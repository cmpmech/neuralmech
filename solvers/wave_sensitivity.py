import numpy as np
import cupy as cp
from pathlib import Path

from solvers.wave import (setup_source, compile_kernels, grid_block,
                          linear_indices, axis_geometry, define_step_method,
                          define_homogeneous_Neumann_BC, define_excitation,
                          define_get_signal)

# Adjoint sensitivity for the scalar (FWI) and acoustic (TATO) wave equations, in two
# variants that share the same Frechet-kernel machinery (Herrmann et al. 2026):
#
#   compute_subtracted_kernels / compute_sensitivity -- the lossless SUPERPOSITION
#     trick (Eqs. 32-39): a single field is propagated, K(u, u) accumulated forward
#     and K(u^s, u^s) backward on the superposed field u^s = u + k u^dagger. Memory-
#     free but limited to self-adjoint problems -- no damping, since it relies on
#     time reversibility.
#
#   compute_sensitivity_pml -- the two-field BOUNDARY-RECONSTRUCTION adjoint: forward
#     field u and adjoint field lambda are propagated separately, so an absorbing
#     sponge layer (see build_sponge in wave.py) is admissible. The lossless interior
#     is reconstructed backward in time while the thin damped strip is stored and
#     replayed; only the strip x number-of-steps is kept, so it scales to 3D / large
#     grids. The sponge breaks reversibility only outside the design region, so the
#     interior gradient is still exact.
#
# Both call the same bilinear integrand kernel (wave_sensitivity.cu): the quadratic
# form K(u, u) is its diagonal case (lambda aliased to u). See wave.cu for the forward
# kernels and the compile-time configuration flags.

SENS_KERNEL_PATH = Path(__file__).parent / "kernels" / "wave_sensitivity.cu"


def sponge_indices(sim, damping):
    # flat indices of every cell carrying damping (the strip to record and replay)
    return cp.where(damping.ravel() > 0)[0].astype(cp.int32)


def define_integrand(sim, kernels):
    integrand_kernel = kernels.get_function("integrand_step_kernel")
    grid, block = grid_block(sim)
    coef_t, coef_grad = sim.frechet_coefficients()
    geom = axis_geometry(sim, [sim.dtype(d) for d in sim.dx])

    def increment_integrand(kernel, u0, u1, u2, l0, l1, l2, scale):
        integrand_kernel(grid, block,
                        (kernel, u0, u1, u2, l0, l1, l2, sim.dtype(sim.dt),
                         sim.dtype(coef_t), sim.dtype(coef_grad), sim.dtype(scale), *geom))
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
    mat = sim.build_materials(indicator)
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, SENS_KERNEL_PATH)

    increment_integrand = define_integrand(sim, sens_kernels)
    adjoint_source = define_adjoint_source(sim, sensors, sens_kernels)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)

    fadjoint = cp.zeros((sim.N, num_sensors), dtype=sim.dtype)
    fadjoint_squared = cp.zeros(num_sensors, dtype=sim.dtype)

    # forward simulation: record adjoint sources and accumulate K(u, u) to subtract
    # (the single field is aliased into both slots of the bilinear kernel)
    for t in range(sim.N):
        u2 = fd_step(u0, u1, u2)
        u2 = excitation_step(u2, source.signal, t)
        u2 = bc_step(u2)
        # adjoint signal is stored in reverse time for the backward pass
        adjoint_source(fadjoint[-(t + 1)], fadjoint_squared, u2, um[t])
        subtracted_kernels = increment_integrand(subtracted_kernels, u0, u1, u2,
                                                 u0, u1, u2, -1.)
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
        subtracted_kernels = increment_integrand(subtracted_kernels, u0, u1, u2,
                                                 u0, u1, u2, 1.)
        u0, u1, u2 = u1, u2, u0

    cost = float(np.prod(sim.dx)) * sim.dt * cp.sum(fadjoint_squared).item()
    return cost, subtracted_kernels


def compute_sensitivity(sim, subtracted_kernels, k_factor, num_sources):
    factor = float(np.prod(sim.dx)) * sim.dt / k_factor / 2. / num_sources
    return subtracted_kernels * sim.dtype(factor)


def _sensitivity_pml_core(sim, design, source, sensors, um, sponge, num_sources,
                          adjoint_scale=None):
    # shared forward+backward machinery for the PML adjoint. adjoint_scale is an
    # optional callable applied after the forward pass: given the per-sensor energies
    # fadjoint_squared it returns a per-sensor factor that rescales each sensor's
    # adjoint source column. Because the adjoint field is linear in its source, this
    # turns the quadratic-tracking gradient into the gradient of any loss L(y) when the
    # factor is 2 dL/dy (see compute_sensitivity_classification). None reproduces the
    # plain tracking adjoint. Returns (cost, gradient, fadjoint_squared).
    strip = sponge_indices(sim, sponge)
    n_strip = int(strip.size)
    num_sensors = sensors.shape[1]

    mat_damped = sim.build_materials(design, sponge)
    mat_lossless = sim.build_materials(design)        # damping defaults to zero
    kernels = compile_kernels(sim)
    sens_kernels = compile_kernels(sim, SENS_KERNEL_PATH)

    fd_damped = define_step_method(sim, kernels, mat_damped)
    fd_lossless = define_step_method(sim, kernels, mat_lossless)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat_damped)
    get_signal = define_get_signal(sim, sensors, kernels)
    increment_integrand = define_integrand(sim, sens_kernels)
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

    # rescale each sensor's adjoint source to match the desired loss (default: none)
    if adjoint_scale is not None:
        fadjoint *= adjoint_scale(fadjoint_squared)[None, :]

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

        kernel = increment_integrand(kernel, r0, r1, r2, l0, l1, l2, 1.)
        r0, r1, r2 = r1, r2, r0
        l0, l1, l2 = l1, l2, l0

    # cost is J = (1/2) integral (u - u_measured)^2; the 1/2 makes it consistent with
    # the adjoint source -(u - u_measured) and hence with the returned gradient (a
    # finite-difference gradient check matches only with this factor). The gradient
    # itself is unchanged, so optimization trajectories and the dB metric are unaffected.
    cost = 0.5 * float(np.prod(sim.dx)) * sim.dt * cp.sum(fadjoint_squared).item()
    gradient = kernel * sim.dtype(float(np.prod(sim.dx)) * sim.dt / num_sources)
    return cost, gradient, fadjoint_squared


def compute_sensitivity_pml(sim, design, source, sensors, um, sponge, num_sources=1):
    # returns (cost, gradient w.r.t. the design field gamma over the padded grid).
    # sponge is a damping field (see build_sponge in wave.py); pass cp.zeros(...) to
    # recover the plain closed-domain adjoint (should match the superposition variant).
    cost, gradient, _ = _sensitivity_pml_core(sim, design, source, sensors, um, sponge,
                                              num_sources)
    return cost, gradient


def compute_sensitivity_classification(sim, design, source, sensors, sponge, label,
                                       num_sources=1):
    # analog-RNN classifier gradient (Hughes et al. 2019): the medium `design` is the
    # trainable weight field, the sensors are one probe per class, and the readout is
    # the integrated probe energy y_m = sum_t u(x_m, t)^2. The prediction is the
    # normalized intensity p = y / sum(y) (equivalently a softmax over log-energies)
    # and the loss is cross-entropy -log p[label]. Running the tracking adjoint against
    # a silent target (um = 0) makes the forward pass yield exactly y (in
    # fadjoint_squared) and an adjoint source -u; rescaling each probe by 2 dL/dy_m then
    # turns the gradient into that of the cross-entropy loss. The prod(dx) dt prefactors
    # cancel in p (raw energies used directly for the readout), but the per-probe adjoint
    # gradient carries a prod(dx) dt factor, so the rescale divides it back out.
    # returns (loss, probs, gradient w.r.t. the design field gamma over the padded grid).
    um = cp.zeros((sim.N, sensors.shape[1]), dtype=sim.dtype)
    prefactor = float(np.prod(sim.dx)) * sim.dt

    def adjoint_scale(y):
        # 2 dL/dy_m with dL/dy_m = 1/sum(y) - delta_{m,label}/y[label]
        scale = cp.full(y.shape, 1.0 / float(cp.sum(y)), dtype=sim.dtype)
        scale[label] -= 1.0 / float(y[label])
        return (2.0 / prefactor) * scale

    _, gradient, y = _sensitivity_pml_core(sim, design, source, sensors, um, sponge,
                                           num_sources, adjoint_scale)
    probs = (y / cp.sum(y)).get()
    loss = float(-np.log(probs[label]))
    return loss, probs, gradient
