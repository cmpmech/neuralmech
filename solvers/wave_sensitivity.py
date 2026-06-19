import numpy as np
import cupy as cp
from pathlib import Path

from solvers.wave import (setup_source, build_materials, compile_kernels, grid_block,
                          linear_indices, define_step_method,
                          define_homogeneous_Neumann_BC, define_excitation)

# Memory-efficient adjoint sensitivity via the superposition of forward and adjoint
# wave fields (Herrmann et al. 2026, Eqs. 32-39). Works for the scalar (FWI) and
# acoustic (TATO) formulations. Limited to self-adjoint problems: damping is not
# admissible because the method relies on time reversibility.

SENS_KERNEL_PATH = Path(__file__).parent / "kernels" / "wave_sensitivity.cu"


def compile_sensitivity_kernels(sim):
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}"]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    if sim.formulation == "acoustic":
        options.append("-DFORMULATION_ACOUSTIC")
    return cp.RawModule(code=SENS_KERNEL_PATH.read_text(), options=tuple(options))


def define_increment_integrand(sim, kernels):
    integrand_kernel = kernels.get_function("integrand_step_kernel")
    grid, block = grid_block(sim)
    coef_t, coef_grad = sim.frechet_coefficients()

    geom = [sim.dtype(sim.dx[0]), sim.Nx[0]]
    for d in range(1, sim.ndim):
        geom += [sim.dtype(sim.dx[d]), sim.Nx[d], sim.strides[d - 1]]

    def increment_integrand(kernel, u0, u1, u2, sign):
        integrand_kernel(grid, block,
                        (kernel, u0, u1, u2, sim.dtype(sim.dt), sim.dtype(coef_t),
                         sim.dtype(coef_grad), sim.dtype(sign), *geom))
        return kernel
    return increment_integrand


def define_adjoint_source(sim, sensors, kernels):
    adjoint_source_kernel = kernels.get_function("adjoint_source_step_kernel")
    threads = 256
    num_sensors = sensors.shape[1]
    blocks = (num_sensors + threads - 1) // threads
    lin_index = linear_indices(sim, sensors)

    def adjoint_source(fadjoint, fadjoint_squared, u, um_t):
        adjoint_source_kernel((blocks,), (threads,),
                              (fadjoint, fadjoint_squared, u, um_t, lin_index, num_sensors))
        return fadjoint, fadjoint_squared
    return adjoint_source


def compute_subtracted_kernels(subtracted_kernels, sim, source, indicator,
                               um, sensors, k_factor):
    U = cp.zeros((3, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1, u2 = U[0], U[1], U[2]
    num_sensors = sensors.shape[1]

    # no damping is admissible: materials are built undamped, keeping the problem
    # self-adjoint and time-reversible as the superposition method requires
    mat = build_materials(sim, indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_sensitivity_kernels(sim)

    increment_integrand = define_increment_integrand(sim, sens_kernels)
    adjoint_source = define_adjoint_source(sim, sensors, sens_kernels)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)

    fadjoint = cp.zeros((sim.N, num_sensors), dtype=sim.dtype)
    fadjoint_squared = cp.zeros(num_sensors, dtype=sim.dtype)

    # forward simulation: record adjoint sources and accumulate K(u, u) to subtract
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2)
        u2 = excitation_step(u2, source.signal, t)
        u2 = bc_step(u2)
        # adjoint signal is stored in reverse time for the backward pass
        adjoint_source(fadjoint[-(t + 1)], fadjoint_squared, u2, um[t])
        subtracted_kernels = increment_integrand(subtracted_kernels, u0, u1, u2, -1.)
        u0, u1, u2 = u1, u2, u0

    fadjoint *= sim.dtype(k_factor)
    adjoint = setup_source(sensors, fadjoint)
    adjoint_excitation = define_excitation(sim, adjoint.position, kernels, mat)

    # backward simulation of the superposed field u^s = u + k u^dagger;
    # accumulate K(u^s, u^s) (reversibility lets us re-integrate without storing u)
    u0, u1 = u1, u0
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2)
        u2 = adjoint_excitation(u2, adjoint.signal, t)
        u2 = excitation_step(u2, source.signal, sim.N - 1 - t)
        u2 = bc_step(u2)
        subtracted_kernels = increment_integrand(subtracted_kernels, u0, u1, u2, 1.)
        u0, u1, u2 = u1, u2, u0

    cost = float(np.prod(sim.dx)) * sim.dt * cp.sum(fadjoint_squared).item()
    return cost, subtracted_kernels


def compute_sensitivity(sim, subtracted_kernels, k_factor, num_sources):
    factor = float(np.prod(sim.dx)) * sim.dt / k_factor / 2. / num_sources
    return subtracted_kernels * sim.dtype(factor)
