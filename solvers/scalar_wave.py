import cupy as cp
import cupy.typing as cpt
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

@dataclass
class setup_source:
    position: cpt.NDArray[cp.int32]
    signal: cpt.NDArray[cp.float32]

@dataclass
class setup_simulation:
    Nx: tuple[int]
    dx: tuple[float]
    N: int
    dt: float
    wavespeed: float
    density: float
    threads : tuple[int]

    def __post_init__(self):
        self.Nx_padded = (self.Nx[0], ((self.Nx[1] + 32 - 1) // 32) * 32)

def define_step_method2D(sim, compiled_kernels):
    fd_kernel = compiled_kernels.get_function('fd_kernel')

    blocks_j = (sim.Nx_padded[1] + sim.threads[1] - 1) // sim.threads[1] # cols (Ny is num cols)
    blocks_i = (sim.Nx_padded[0] + sim.threads[0] - 1) // sim.threads[0] # rows (Nx is num rows)

    # precomputed
    laplace_factor_x = cp.float32(2. * sim.wavespeed ** 2 * sim.dt ** 2 / sim.dx[0] ** 2)
    laplace_factor_y = cp.float32(2. * sim.wavespeed ** 2 * sim.dt ** 2 / sim.dx[1] ** 2)

    def fd_step(u0, u1, u2, indicator):
        fd_kernel((blocks_j, blocks_i), (sim.threads[1], sim.threads[0]),
                  (u0, u1, u2, indicator, laplace_factor_x,
                   laplace_factor_y, sim.Nx[0], sim.Nx[1], sim.Nx_padded[1]))
        return u2
    return fd_step

def define_homogeneous_Neumann_BC2D(sim, compiled_kernels):
    bc_kernel = compiled_kernels.get_function('bc_kernel')

    threads = 256
    # boundary points excluding corners
    top_bottom = 2 * (sim.Nx[1] - 2)  # two horizontal edges
    left_right = 2 * (sim.Nx[0] - 2)  # two vertical edges
    perimeter = top_bottom + left_right
    blocks = (perimeter + threads - 1) // threads

    def bc_method(u):
        bc_kernel((blocks,), (threads,),
                       (u, sim.Nx[0], sim.Nx[1],
                        sim.Nx_padded[1]))
        return u
    return bc_method

def define_excitation2D(sim, num_sources, compiled_kernels):
    excitation_kernel = compiled_kernels.get_function('excitation_kernel')

    threads = 256
    blocks = (num_sources + threads - 1) // threads

    # precomputed
    dt2density = cp.float32(sim.dt**2 / sim.density)

    def excitation_method(u, signal, position, t_index, indicator):
        excitation_kernel((blocks,), (threads,),
                          (u, signal[t_index], position, len(position[0]),
                                dt2density, indicator, sim.Nx_padded[1]))
        return u
    return excitation_method

def define_get_signal2D(sim, sensors, compiled_kernels):
    get_signal_kernel = compiled_kernels.get_function('get_signal_kernel')

    threads = 256
    num_sensors = len(sensors[0])
    blocks = (num_sensors + threads - 1) // threads

    def get_signal_method(u, um):
        get_signal_kernel((blocks,), (threads,),
                          (u, um, sensors[0], sensors[1], num_sensors,
                           sim.Nx_padded[1]))
        return um
    return get_signal_method

def simulate2D(sim, source, indicator, sensors=None, precompiled=False):
    U = cp.zeros((2, *sim.Nx_padded), dtype=cp.float32)
    u0 = U[0]
    u1 = U[1]

    # stepping methods
    if precompiled:
        compiled_kernels = cp.RawModule(path=str(Path(__file__).parent/'kernels_build'/'scalar_wave.ptx'))
    else:
        compiler_options = ('--use_fast_math',)
        compiled_kernels = cp.RawModule(code=(Path(__file__).parent/'kernels'/'scalar_wave.cu').read_text(),
                                        options=compiler_options)
    fd_step = define_step_method2D(sim, compiled_kernels)
    bc_step = define_homogeneous_Neumann_BC2D(sim, compiled_kernels)
    excitation_step = define_excitation2D(sim, len(source.position[0]), compiled_kernels)
    if sensors is not None:
        get_signal = define_get_signal2D(sim, sensors, compiled_kernels)
        um = cp.zeros((sim.N, len(sensors[0])), dtype=cp.float32)

    for t in tqdm(range(sim.N)):
        u0 = fd_step(u0, u1, u0, indicator)
        u0 = excitation_step(u0, source.signal, source.position, t, indicator)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
        if sensors is not None:
            um[t] = get_signal(u1, um[t])

    if sensors is not None:
        return u1[:sim.Nx[0], :sim.Nx[1]], um
    else:
        return u1[:sim.Nx[0], :sim.Nx[1]]