from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import cupy.typing as cpt
import numpy as np
from tqdm import tqdm

KERNEL_PATH = Path(__file__).parent / "kernels" / "wave.cu"


@dataclass
class setup_source:
    position: cpt.NDArray[cp.int32]  # (ndim, num_sources) grid indices
    signal: cpt.NDArray  # (N, num_sources) time series
    component: int = 0  # displacement component driven (vector formulations only)


@dataclass
class setup_simulation:
    # shared grid / time / precision scaffolding; the formulation-specific material
    # model lives in the scalar_simulation / acoustic_simulation subclasses below
    Nx: tuple[int, ...]  # logical grid points per axis (incl. ghosts)
    dx: tuple[float, ...]
    N: int  # number of time steps
    dt: float
    threads: tuple[int, ...]  # threads per block, per axis
    precision: str = "float32"  # "float32" or "float64"

    compile_flags = ()  # extra nvcc -D flags for compile_kernels
    ncomp = 1  # unknowns per grid point (scalar field); elastic overrides to ndim

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
        self.comp_stride = int(np.prod(self.Nx_padded))  # offset between components


@dataclass
class scalar_simulation(setup_simulation):
    wavespeed: float = None  # background wave speed c0
    density: float = None  # background density rho0

    def __post_init__(self):
        super().__post_init__()
        if self.wavespeed is None or self.density is None:
            raise ValueError("scalar_simulation requires wavespeed and density")

    def frechet_coefficients(self):
        return -self.density, self.density * self.wavespeed**2

    def build_materials(self, indicator, damping=None):
        # indicator is the rho-scaling field gamma
        return {"gamma": indicator}

    def step_factors(self):
        return [self.dtype(2.0 * self.wavespeed**2 * self.dt**2 / dxk**2) for dxk in self.dx]

    def step_kernel_args(self, mat):
        return (mat["gamma"],)

    def excitation_params(self, mat):
        return mat["gamma"], self.dtype(self.dt**2 / self.density)  # dt^2 / (rho0 gamma)


@dataclass
class acoustic_simulation(setup_simulation):
    # TATO material constants (gamma = 0 -> air, gamma = 1 -> solid)
    rho1: float = None
    rho2: float = None
    kappa1: float = None
    kappa2: float = None

    compile_flags = ("-DFORMULATION_ACOUSTIC",)

    def __post_init__(self):
        super().__post_init__()
        if None in (self.rho1, self.rho2, self.kappa1, self.kappa2):
            raise ValueError("acoustic_simulation requires rho1, rho2, kappa1, kappa2")

    def frechet_coefficients(self):
        return -(1 / self.kappa2 - 1 / self.kappa1), (1 / self.rho2 - 1 / self.rho1)

    def build_materials(self, indicator, damping=None):
        # gamma interpolates the inverse density and inverse bulk modulus
        rho_inv = 1 / self.rho1 + indicator * (1 / self.rho2 - 1 / self.rho1)
        kappa_inv = 1 / self.kappa1 + indicator * (1 / self.kappa2 - 1 / self.kappa1)
        if damping is None:
            damping = cp.zeros(self.Nx_padded, dtype=self.dtype)
        return {"rho": 1 / rho_inv, "kappa": 1 / kappa_inv, "damping": damping}

    def step_factors(self):
        return [self.dtype(2.0 * self.dt**2 / dxk**2) for dxk in self.dx]  # kappa in-kernel

    def step_kernel_args(self, mat):
        return (mat["rho"], mat["kappa"], mat["damping"], self.dtype(self.dt))

    def excitation_params(self, mat):
        return mat["kappa"], self.dtype(self.dt**2)  # kappa dt^2


@dataclass
class elastic_simulation(setup_simulation):
    # isotropic elastodynamics rho u_tt = div(gamma sigma), sigma = lam tr(eps) I
    # + 2 mu eps; the design field gamma scales inertia and stiffness together, so
    # a uniform gamma cancels and only its gradients scatter (like the scalar case)
    density: float = None  # background density rho0
    lame_lambda: float = None  # first Lame parameter
    lame_mu: float = None  # second Lame parameter (shear modulus)

    compile_flags = ("-DFORMULATION_ELASTIC",)

    def __post_init__(self):
        super().__post_init__()
        self.ncomp = self.ndim
        if None in (self.density, self.lame_lambda, self.lame_mu):
            raise ValueError("elastic_simulation requires density, lame_lambda, lame_mu")
        if self.ndim not in (2, 3):
            raise ValueError("elastic_simulation supports NDIM in {2, 3}")

    def frechet_coefficients(self):
        return -self.density, self.lame_lambda, self.lame_mu  # coef_t, lam, mu

    def build_materials(self, indicator, damping=None):
        return {"gamma": indicator}

    def step_factors(self):
        return [self.dtype(1.0 / dxk) for dxk in self.dx]  # inverse spacing per axis

    def step_kernel_args(self, mat):
        return (mat["gamma"], np.int32(self.comp_stride), self.dtype(self.lame_lambda),
                self.dtype(self.lame_mu), self.dtype(self.density), self.dtype(self.dt**2))

    def excitation_params(self, mat):
        return mat["gamma"], self.dtype(self.dt**2 / self.density)  # dt^2 / (rho gamma)


def compile_kernels(sim, path=KERNEL_PATH):
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}", *sim.compile_flags]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    return cp.RawModule(code=Path(path).read_text(), options=tuple(options))


def build_sponge(sim, width, d_max, power=3, sides=None):
    # absorbing strip: damping ramps from 0 at the interface to d_max at the outer
    # face as (depth / width)^power. sides is a list of (axis, end) with end in
    # {"lo", "hi"}; default absorbs both ends of the last axis (top / bottom). The
    # resulting field feeds build_materials / simulate as the damping argument.
    if sides is None:
        sides = [(sim.ndim - 1, "lo"), (sim.ndim - 1, "hi")]
    damping = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

    for axis, end in sides:
        n = sim.Nx[axis]
        ramp = cp.zeros(n, dtype=sim.dtype)
        depth = cp.arange(width, dtype=sim.dtype)
        profile = d_max * ((width - depth) / width) ** power
        if end == "lo":
            ramp[1 : 1 + width] = profile           # cells just inside the ghost layer
        else:
            ramp[n - 1 - width : n - 1] = profile[::-1]
        shape = [1] * sim.ndim
        shape[axis] = n
        sl = tuple(slice(0, s) for s in sim.Nx)     # write into the logical interior
        damping[sl] = cp.maximum(damping[sl], ramp.reshape(shape)[sl])
    return damping


def grid_block(sim):
    # map the fastest axis to grid/block x, the next to y, the next to z
    extent = sim.Nx_padded
    block = tuple(sim.threads[::-1])
    grid = tuple(
        (extent[d] + sim.threads[d] - 1) // sim.threads[d] for d in range(sim.ndim)
    )[::-1]
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
    geom = axis_geometry(sim, sim.step_factors())
    args = sim.step_kernel_args(mat)

    def fd_step(u0, u1, u2):
        fd_kernel(grid, block, (u0, u1, u2, *args, *geom))
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

    # each displacement component is mirrored independently: component-wise
    # homogeneous Neumann is the simple reflective wall for the vector formulation
    def bc_step(u):
        for c in range(sim.ncomp):
            uc = u[c]
            for axis, grid, block in launches:
                bc_kernel(grid, block, (uc, axis, *geom))
        return u

    return bc_step


def linear_indices(sim, position):
    # collapse (ndim, num) grid indices into flat indices of the padded array
    lin = cp.zeros(position.shape[1], dtype=cp.int32)
    for d in range(sim.ndim):
        lin += position[d] * cp.int32(sim.strides[d])
    return lin


def define_excitation(sim, position, kernels, mat, component=0):
    excitation_kernel = kernels.get_function("excitation_kernel")
    threads = 256
    num_sources = position.shape[1]
    blocks = (num_sources + threads - 1) // threads
    lin_index = linear_indices(sim, position)
    scale, dt2 = sim.excitation_params(mat)

    def excitation_step(u, signal, t_index):
        excitation_kernel(
            (blocks,),
            (threads,),
            (u[component], signal[t_index], lin_index, num_sources, dt2, scale),
        )
        return u

    return excitation_step


def define_get_signal(sim, sensors, kernels, component=0):
    get_signal_kernel = kernels.get_function("get_signal_kernel")
    threads = 256
    num_sensors = sensors.shape[1]
    blocks = (num_sensors + threads - 1) // threads
    lin_index = linear_indices(sim, sensors)

    def get_signal_step(u, um_t):
        get_signal_kernel((blocks,), (threads,), (u[component], um_t, lin_index, num_sensors))
        return um_t

    return get_signal_step


def simulate(sim, source, indicator, damping=None, sensors=None, record_every=None,
             sensor_component=0):
    # field carries sim.ncomp components (1 for scalar/acoustic, ndim for elastic);
    # the component axis collapses transparently so scalar callers see a plain field
    U = cp.zeros((2, sim.ncomp, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1 = U[0], U[1]

    mat = sim.build_materials(indicator, damping)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat, source.component)
    if sensors is not None:
        get_signal = define_get_signal(sim, sensors, kernels, sensor_component)
        um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    interior = tuple(slice(0, n) for n in sim.Nx)
    field = lambda u: u[0][interior] if sim.ncomp == 1 else u[(slice(None), *interior)]
    snapshots = []

    for t in tqdm(range(sim.N)):
        u0 = fd_step(u0, u1, u0)
        u0 = excitation_step(u0, source.signal, t)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
        if sensors is not None:
            um[t] = get_signal(u1, um[t])
        if record_every is not None and t % record_every == 0:
            snapshots.append(field(u1).get())

    if sensors is not None and record_every is not None:
        return field(u1), um, np.stack(snapshots)
    if sensors is not None:
        return field(u1), um
    if record_every is not None:
        return field(u1), np.stack(snapshots)
    return field(u1)
