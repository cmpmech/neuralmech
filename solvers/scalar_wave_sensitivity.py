import numpy as np
import cupy as cp
from setup_scalar import (setup_source, define_step_method2D,
                          define_homogeneous_Neumann_BC2D, define_excitation2D)
from pathlib import Path

def define_increment_integrand2D(sim, compiled_kernels):
    integrand_kernel = compiled_kernels.get_function('integrand_step_kernel')
    blocks_j = (sim.Nx_padded[1] + sim.threads[1] - 1) // sim.threads[1]  # cols (Ny is num cols)
    blocks_i = (sim.Nx_padded[0] + sim.threads[0] - 1) // sim.threads[0]

    # precomputed
    densitywavespeed2 = cp.float32(sim.density * sim.wavespeed**2)

    def increment_integrand_method(kernel, u0, u1, u2, sign):
        integrand_kernel((blocks_j, blocks_i), (sim.threads[1], sim.threads[0]),
                         (kernel, u0, u1, u2, cp.float32(sim.dt), cp.float32(sim.dx[0]),
                          cp.float32(sim.dx[1]), cp.float32(sim.density), densitywavespeed2,
                          cp.float32(sign), sim.Nx[0], sim.Nx[1], sim.Nx_padded[1]))
        return kernel
    return increment_integrand_method

def define_adjoint_source2D(sim, sensors, compiled_kernels):
    adjoint_source_kernel = compiled_kernels.get_function('adjoint_source_step_kernel')
    threads = 256
    num_sensors = len(sensors[0])
    blocks = (num_sensors + threads - 1) // threads

    def adjoint_source_method(fadjoint, fadjoint_squared, u, um, t):
        adjoint_source_kernel((blocks,), (threads,),
                              (fadjoint[-(t + 1)], fadjoint_squared, u, um[t],
                               sensors, num_sensors, sim.Nx_padded[1]))
        return fadjoint, fadjoint_squared
    return adjoint_source_method


def compute_substracted_kernels2D(subtracted_kernels, sim, source,
                                  indicator, um, sensors, k_factor, precompiled=False):
    U = cp.zeros((3, *sim.Nx_padded), dtype=cp.float32)
    u0, u1, u2 = U[0], U[1], U[2]

    num_sensors = len(sensors[0])

    if precompiled:
        compiled_kernels_sensitivity = cp.RawModule(path=str(Path(__file__).parent/'kernels_build'/'scalar_wave_sensitivity.ptx'))
        compiled_kernels_forward = cp.RawModule(path=str(Path(__file__).parent/'kernels_build'/'scalar_wave.ptx'))
    else:
        compiler_options = ('--use_fast_math',)
        compiled_kernels_sensitivity = cp.RawModule(code=(Path(__file__).parent/'kernels'/'scalar_wave_sensitivity.cu').read_text(),
                                                    options=compiler_options)
        compiled_kernels_forward = cp.RawModule(code=(Path(__file__).parent/'kernels'/'scalar_wave.cu').read_text(),
                                                options=compiler_options)

    increment_integrand = define_increment_integrand2D(sim, compiled_kernels_sensitivity)
    adjoint_source = define_adjoint_source2D(sim, sensors, compiled_kernels_sensitivity)
    fd_step = define_step_method2D(sim, compiled_kernels_forward)
    bc_step = define_homogeneous_Neumann_BC2D(sim, compiled_kernels_forward)
    excitation_step = define_excitation2D(sim, num_sensors, compiled_kernels_forward)

    # initialization
    fadjoint = cp.zeros((sim.N, num_sensors), dtype=cp.float32)
    fadjoint_squared = cp.zeros((num_sensors), dtype=cp.float32)

    # forward simulation
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2, indicator)
        u2 = excitation_step(u2, source.signal, source.position, t, indicator)
        u2 = bc_step(u2)

        fadjoint, fadjoint_squared = adjoint_source(fadjoint, fadjoint_squared, u2, um, t) # TODO is this really u1 ?
        subtracted_kernels = increment_integrand(subtracted_kernels, u0, u1, u2, -1.)

        u0, u1, u2 = u1, u2, u0

    fadjoint *= cp.float32(k_factor)
    adjoint_source = setup_source(sensors, fadjoint)

    # backward simulation
    u0, u1 = u1, u0

    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2, indicator)
        # superpose forces
        u2 = excitation_step(u2, adjoint_source.signal, adjoint_source.position, t, indicator)
        u2 = excitation_step(u2, source.signal, source.position, sim.N - 1 - t, indicator) # index is flipped
        u2 = bc_step(u2)
        subtracted_kernels = increment_integrand(subtracted_kernels, u0, u1, u2, 1.) # computes superposed kernel

        u0, u1, u2 = u1, u2, u0

    cost = sim.dx[0] * sim.dx[1] * sim.dt * cp.sum(fadjoint_squared).item()
    return cost, subtracted_kernels

def compute_sensitivity(sim, subtracted_kernels, k_factor, num_sources):
    factor = np.prod(sim.dx) * sim.dt / k_factor / 2. / num_sources
    return subtracted_kernels * cp.float32(factor)