import numpy as np
import cupy as cp
from pathlib import Path

from solvers.wave import (setup_source, build_materials, compile_kernels, grid_block,
                          linear_indices, axis_geometry, define_step_method,
                          define_homogeneous_Neumann_BC, define_excitation,
                          define_get_signal)

# Two-field boundary-reconstruction adjoint sensitivity. Unlike wave_sensitivity.py
# (the lossless superposition trick), this variant admits an absorbing sponge layer:
# the interior is kept lossless and reconstructed backward in time, while the thin
# damped strip is stored over time and replayed. The forward field u and adjoint
# field lambda are propagated as two separate fields, so the Frechet kernel is the
# bilinear form K(u, lambda) (see wave_sensitivity_pml.cu). Memory scales with the
# strip area times the number of steps, not the full volume, so it moves to 3D / large
# grids. The sponge breaks reversibility only in the strip, which lies outside the
# design region, so the gradient in the interior is still exact.

SENS_KERNEL_PATH = Path(__file__).parent / "kernels" / "wave_sensitivity_pml.cu"


def build_sponge(sim, width, d_max, power=3, sides=None):
    # absorbing strip: damping ramps from 0 at the interface to d_max at the outer
    # face as (depth / width)^power. sides is a list of (axis, end) with end in
    # {"lo", "hi"}; default absorbs both ends of the last axis (top / bottom).
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


def sponge_indices(sim, damping):
    # flat indices of every cell carrying damping (the strip to record and replay)
    return cp.where(damping.ravel() > 0)[0].astype(cp.int32)


def compile_sensitivity_kernels(sim):
    options = ["--use_fast_math", f"-DNDIM={sim.ndim}"]
    if sim.precision == "float32":
        options.append("-DUSE_FLOAT")
    if sim.formulation == "acoustic":
        options.append("-DFORMULATION_ACOUSTIC")
    return cp.RawModule(code=SENS_KERNEL_PATH.read_text(), options=tuple(options))


def define_bilinear_integrand(sim, kernels):
    integrand_kernel = kernels.get_function("bilinear_integrand_step_kernel")
    grid, block = grid_block(sim)
    coef_t, coef_grad = sim.frechet_coefficients()
    geom = axis_geometry(sim, [sim.dtype(d) for d in sim.dx])

    def increment_integrand(kernel, u0, u1, u2, l0, l1, l2):
        integrand_kernel(grid, block,
                        (kernel, u0, u1, u2, l0, l1, l2, sim.dtype(sim.dt),
                         sim.dtype(coef_t), sim.dtype(coef_grad), *geom))
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


def compute_sensitivity_pml(sim, design, source, sensors, um, sponge, num_sources=1):
    # returns (cost, gradient w.r.t. the design field gamma over the padded grid).
    # sponge is a damping field (see build_sponge); pass cp.zeros(...) to recover the
    # plain closed-domain adjoint (should match the superposition variant).
    strip = sponge_indices(sim, sponge)
    n_strip = int(strip.size)
    num_sensors = sensors.shape[1]

    mat_damped = build_materials(sim, design, sponge)
    mat_lossless = build_materials(sim, design)        # damping defaults to zero
    kernels = compile_kernels(sim)
    sens_kernels = compile_sensitivity_kernels(sim)

    fd_damped = define_step_method(sim, kernels, mat_damped)
    fd_lossless = define_step_method(sim, kernels, mat_lossless)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat_damped)
    get_signal = define_get_signal(sim, sensors, kernels)
    increment_integrand = define_bilinear_integrand(sim, sens_kernels)
    adjoint_source = define_adjoint_source(sim, sensors, sens_kernels)

    # forward pass: damped sim, record the strip over time + the final two snapshots,
    # the per-step sensor residuals (reversed for the adjoint), and the cost
    U = cp.zeros((3, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1, u2 = U[0], U[1], U[2]
    strip_store = cp.zeros((sim.N, n_strip), dtype=sim.dtype)
    fadjoint = cp.zeros((sim.N, num_sensors), dtype=sim.dtype)
    fadjoint_squared = cp.zeros(num_sensors, dtype=sim.dtype)
    um_t = cp.zeros(num_sensors, dtype=sim.dtype)
    seed_last = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    seed_prev = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

    for t in range(sim.N):
        u2 = fd_damped(u0, u1, u2)
        u2 = excitation_step(u2, source.signal, t)
        u2 = bc_step(u2)
        strip_store[t] = u2.ravel()[strip]
        um_t = get_signal(u2, um_t)
        # residual stored in reverse time for the backward adjoint pass
        adjoint_source(fadjoint[-(t + 1)], fadjoint_squared, u2, um[t])
        if t == sim.N - 2:
            seed_prev[...] = u2
        if t == sim.N - 1:
            seed_last[...] = u2
        u0, u1, u2 = u1, u2, u0

    adjoint = setup_source(sensors, fadjoint)
    adjoint_excitation = define_excitation(sim, adjoint.position, kernels, mat_damped)

    # backward pass: reconstruct the lossless interior (source re-injected in reverse
    # time, strip replayed from storage) while propagating the absorbed adjoint field,
    # accumulating the bilinear Frechet kernel K(u, lambda)
    R = cp.zeros((3, *sim.Nx_padded), dtype=sim.dtype)
    r0, r1, r2 = R[0], R[1], R[2]
    r0[...] = seed_last                                # u at t = N - 1
    r1[...] = seed_prev                                # u at t = N - 2
    L = cp.zeros((3, *sim.Nx_padded), dtype=sim.dtype)
    l0, l1, l2 = L[0], L[1], L[2]

    kernel = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    for t in range(sim.N):
        # reverse leapfrog: r2 becomes u at physical time m = N - 3 - t
        r2 = fd_lossless(r0, r1, r2)
        r2 = excitation_step(r2, source.signal, sim.N - 1 - t)
        m = sim.N - 3 - t
        if m >= 0:
            r2.ravel()[strip] = strip_store[m]         # replay the damped boundary
        r2 = bc_step(r2)

        l2 = fd_damped(l0, l1, l2)
        l2 = adjoint_excitation(l2, adjoint.signal, t)
        l2 = bc_step(l2)

        kernel = increment_integrand(kernel, r0, r1, r2, l0, l1, l2)
        r0, r1, r2 = r1, r2, r0
        l0, l1, l2 = l1, l2, l0

    # cost is J = (1/2) integral (u - u_measured)^2; the 1/2 makes it consistent with
    # the adjoint source -(u - u_measured) and hence with the returned gradient (a
    # finite-difference gradient check matches only with this factor). The gradient
    # itself is unchanged, so optimization trajectories and the dB metric are unaffected.
    cost = 0.5 * float(np.prod(sim.dx)) * sim.dt * cp.sum(fadjoint_squared).item()
    gradient = kernel * sim.dtype(float(np.prod(sim.dx)) * sim.dt / num_sources)
    return cost, gradient
