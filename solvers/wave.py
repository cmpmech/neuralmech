from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
from tqdm import tqdm

KERNEL_PATH = Path(__file__).parent / "kernels" / "wave.cu"


@dataclass
class setup_source:
    position: cpt.NDArray[cp.int32]   # (ndim, num_sources) grid indices
    signal: cpt.NDArray               # (N, num_sources) time series


@dataclass
class setup_simulation:
    Nx: tuple[int, ...]               # logical grid points per axis (incl. ghosts)
    dx: tuple[float, ...]
    N: int                            # number of time steps
    dt: float
    wavespeed: float                  # scalar: background wave speed c0
    density: float                    # scalar: background density rho0
    threads: tuple[int, ...]          # threads per block, per axis
    formulation: str = "scalar"       # "scalar" or "acoustic"
    precision: str = "float32"        # "float32" or "float64"
    # acoustic TATO material constants (gamma = 0 -> air, gamma = 1 -> solid)
    rho1: float = 1.204
    rho2: float = 2643.
    kappa1: float = 1.419e5
    kappa2: float = 6.87e8

    def __post_init__(self):
        self.ndim = len(self.Nx)
        # pad the fastest (last) axis to a multiple of 32 for coalesced access
        self.Nx_padded = (*self.Nx[:-1], ((self.Nx[-1] + 31) // 32) * 32)
        # C-contiguous strides over the padded shape (last axis has unit stride)
        strides = [1] * self.ndim
        for d in range(self.ndim - 2, -1, -1):
            strides[d] = strides[d + 1] * self.Nx_padded[d + 1]
        self.strides = tuple(strides)
        self.dtype = cp.float32 if self.precision == "float32" else cp.float64

    def frechet_coefficients(self):
        # (coef_t, coef_grad) of the Frechet kernel  coef_t u'_t u_t + coef_grad grad u' . grad u
        if self.formulation == "scalar":
            return -self.density, self.density * self.wavespeed ** 2      # Eq. 10
        return -(1 / self.kappa2 - 1 / self.kappa1), (1 / self.rho2 - 1 / self.rho1)  # Eq. 20


def build_materials(sim, indicator, damping=None):
    # scalar: indicator is the rho-scaling field gamma. acoustic: gamma interpolates
    # the inverse density and inverse bulk modulus (Eq. 17-18); the scheme uses the
    # density rho and bulk modulus kappa.
    if sim.formulation == "scalar":
        return {"gamma": indicator}

    rho_inv = 1 / sim.rho1 + indicator * (1 / sim.rho2 - 1 / sim.rho1)
    kappa_inv = 1 / sim.kappa1 + indicator * (1 / sim.kappa2 - 1 / sim.kappa1)
    if damping is None:
        damping = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    return {"rho": 1 / rho_inv, "kappa": 1 / kappa_inv, "damping": damping}


def compile_kernels(sim):
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}"]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    if sim.formulation == "acoustic":
        options.append("-DFORMULATION_ACOUSTIC")
    return cp.RawModule(code=KERNEL_PATH.read_text(), options=tuple(options))


def grid_block(sim):
    # map the fastest axis to grid/block x, the next to y, the next to z
    extent = sim.Nx_padded
    block = tuple(sim.threads[::-1])
    grid = tuple((extent[d] + sim.threads[d] - 1) // sim.threads[d]
                 for d in range(sim.ndim))[::-1]
    return grid, block


def axis_geometry(sim, factors):
    # kernel args after the material arrays: f0, N0, [f1, N1, s0], [f2, N2, s1]
    geom = [factors[0], sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [factors[d], sim.Nx[d], sim.strides[d - 1]]
    return geom


def define_step_method(sim, kernels, mat):
    fd_kernel = kernels.get_function("fd_kernel")
    grid, block = grid_block(sim)
    dt2 = sim.dt ** 2

    if sim.formulation == "scalar":
        factors = [sim.dtype(2. * sim.wavespeed ** 2 * dt2 / dxk ** 2) for dxk in sim.dx]
        geom = axis_geometry(sim, factors)
        gamma = mat["gamma"]

        def fd_step(u0, u1, u2):
            fd_kernel(grid, block, (u0, u1, u2, gamma, *geom))
            return u2
    else:
        factors = [sim.dtype(2. * dt2 / dxk ** 2) for dxk in sim.dx]   # kappa applied in-kernel
        geom = axis_geometry(sim, factors)
        rho, kappa, damping = mat["rho"], mat["kappa"], mat["damping"]

        def fd_step(u0, u1, u2):
            fd_kernel(grid, block, (u0, u1, u2, rho, kappa, damping,
                                    sim.dtype(sim.dt), *geom))
            return u2
    return fd_step


def define_homogeneous_Neumann_BC(sim, kernels):
    bc_kernel = kernels.get_function("bc_kernel")
    threads = 256

    launches = []
    for axis in range(sim.ndim):
        other = [d for d in range(sim.ndim) if d != axis]
        face = [sim.Nx[d] - 2 for d in other]
        if sim.ndim == 1:
            grid, block = (1,), (1,)
        elif sim.ndim == 2:
            grid, block = ((face[0] + threads - 1) // threads,), (threads,)
        else:
            # kernel maps threadIdx.x to the last "other" axis, .y to the first
            tb = (16, 16)
            grid = ((face[1] + tb[0] - 1) // tb[0], (face[0] + tb[1] - 1) // tb[1])
            block = tb
        launches.append((axis, grid, block))

    geom = [sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [sim.Nx[d], sim.strides[d - 1]]

    def bc_step(u):
        for axis, grid, block in launches:
            bc_kernel(grid, block, (u, axis, *geom))
        return u
    return bc_step


def linear_indices(sim, position):
    # collapse (ndim, num) grid indices into flat indices of the padded array
    lin = cp.zeros(position.shape[1], dtype=cp.int32)
    for d in range(sim.ndim):
        lin += position[d] * cp.int32(sim.strides[d])
    return lin


def define_excitation(sim, position, kernels, mat):
    excitation_kernel = kernels.get_function("excitation_kernel")
    threads = 256
    num_sources = position.shape[1]
    blocks = (num_sources + threads - 1) // threads
    lin_index = linear_indices(sim, position)

    if sim.formulation == "scalar":
        scale = mat["gamma"]
        dt2 = sim.dtype(sim.dt ** 2 / sim.density)   # source term dt^2 / (rho0 gamma)
    else:
        scale = mat["kappa"]
        dt2 = sim.dtype(sim.dt ** 2)                 # source term kappa dt^2

    def excitation_step(u, signal, t_index):
        excitation_kernel((blocks,), (threads,),
                          (u, signal[t_index], lin_index, num_sources, dt2, scale))
        return u
    return excitation_step


def define_get_signal(sim, sensors, kernels):
    get_signal_kernel = kernels.get_function("get_signal_kernel")
    threads = 256
    num_sensors = sensors.shape[1]
    blocks = (num_sensors + threads - 1) // threads
    lin_index = linear_indices(sim, sensors)

    def get_signal_step(u, um_t):
        get_signal_kernel((blocks,), (threads,), (u, um_t, lin_index, num_sensors))
        return um_t
    return get_signal_step


def simulate(sim, source, indicator, damping=None, sensors=None):
    U = cp.zeros((2, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1 = U[0], U[1]

    mat = build_materials(sim, indicator, damping)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    if sensors is not None:
        get_signal = define_get_signal(sim, sensors, kernels)
        um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)

    for t in tqdm(range(sim.N)):
        u0 = fd_step(u0, u1, u0)
        u0 = excitation_step(u0, source.signal, t)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
        if sensors is not None:
            um[t] = get_signal(u1, um[t])

    interior = tuple(slice(0, n) for n in sim.Nx)
    if sensors is not None:
        return u1[interior], um
    return u1[interior]
