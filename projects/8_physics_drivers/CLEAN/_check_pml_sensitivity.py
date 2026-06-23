import math
import numpy as np
import cupy as cp

from solvers.wave import setup_simulation, setup_source
from solvers.wave_sensitivity import compute_subtracted_kernels, compute_sensitivity
from solvers.wave_sensitivity_pml import compute_sensitivity_pml, build_sponge

# Verifies the boundary-reconstruction (PML) adjoint sensitivity:
#   check 1  PML gradient == the trusted superposition gradient in the lossless limit
#   check 2  directional finite-difference gradient check (the quantity optimization
#            uses), at moderate contrast, showing convergence under grid refinement.
#            A continuous-adjoint gradient cannot match a *single-cell* discrete FD at
#            high contrast on a coarse grid (optimize-then-discretize gap), so the
#            honest check is directional + refined.
#   check 3  sponge absorption quality.

rng0 = np.random.default_rng(0)


def build_case(Nx, N, contrast):
    LENGTHS = (4.0, 4.0)
    dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(2))
    RHO1 = 1.204
    KAPPA1 = 1.419e5
    RHO2 = RHO1 * contrast
    KAPPA2 = KAPPA1 * contrast
    g = cp.linspace(0, 1, 21)
    rho_inv = 1 / RHO1 + g * (1 / RHO2 - 1 / RHO1)
    kappa_inv = 1 / KAPPA1 + g * (1 / KAPPA2 - 1 / KAPPA1)
    wavespeed = float(cp.max(cp.sqrt(rho_inv / kappa_inv)))
    dt = 0.4 * min(dx) / wavespeed / math.sqrt(2)
    sim = setup_simulation(Nx, dx, N, dt, wavespeed, RHO1, (16, 16),
                           formulation="acoustic", precision="float64",
                           rho1=RHO1, rho2=RHO2, kappa1=KAPPA1, kappa2=KAPPA2)
    t_arr = np.linspace(0, (N - 1) * dt, N)
    sig = 1e3 * np.sin(2 * np.pi * 60 * t_arr) * (t_arr < 3 / 60)
    signal = cp.asarray(sig[:, None], dtype=sim.dtype)
    src_i = max(4, Nx[0] // 7)
    source = setup_source(cp.array([[src_i], [Nx[1] // 2]], dtype=cp.int32), signal)
    s0 = (int(Nx[0] * 0.75), int(Nx[1] * 0.45))
    bi, bj = cp.meshgrid(cp.arange(s0[0], s0[0] + 4), cp.arange(s0[1], s0[1] + 4), indexing="ij")
    sensors = cp.stack([bi.ravel(), bj.ravel()]).astype(cp.int32)
    um = cp.asarray(rng0.standard_normal((N, sensors.shape[1])) * 1e-6, dtype=sim.dtype)
    return sim, source, sensors, um, (RHO1, KAPPA1)


def directional_relerr(Nx, N, contrast):
    sim, source, sensors, um, (RHO1, KAPPA1) = build_case(Nx, N, contrast)
    design = cp.full(sim.Nx_padded, 0.5, dtype=sim.dtype)
    zero = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    _, grad = compute_sensitivity_pml(sim, design, source, sensors, um, zero)
    # smooth perturbation in the interior
    lo0, hi0 = Nx[0] // 4, 3 * Nx[0] // 4
    lo1, hi1 = Nx[1] // 4, 3 * Nx[1] // 4
    v = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    v[lo0:hi0, lo1:hi1] = cp.asarray(
        np.random.default_rng(7).standard_normal((hi0 - lo0, hi1 - lo1)), dtype=sim.dtype)
    for axis in (0, 1):
        v = (v + cp.roll(v, 1, axis) + cp.roll(v, -1, axis)) / 3
    mask = cp.zeros_like(v); mask[lo0 + 1:hi0 - 1, lo1 + 1:hi1 - 1] = 1; v *= mask
    eps = 1e-4
    cP, _ = compute_sensitivity_pml(sim, design + eps * v, source, sensors, um, zero)
    cM, _ = compute_sensitivity_pml(sim, design - eps * v, source, sensors, um, zero)
    fd = (cP - cM) / (2 * eps)
    an = float(cp.sum(grad * v))
    return abs(an - fd) / abs(fd), an, fd


# ---------------------------------------------------------------------------
# check 1: lossless limit -> PML gradient must match the superposition gradient
sim, source, sensors, um, _ = build_case((40, 40), 120, 2643.0 / 1.204)
design = cp.asarray(rng0.random(sim.Nx_padded) * 0.5 + 0.25, dtype=sim.dtype)
zero = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
_, grad_pml = compute_sensitivity_pml(sim, design, source, sensors, um, zero)
sk = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
_, sk = compute_subtracted_kernels(sk, sim, source, design, um, sensors, 1e-2)
grad_sup = compute_sensitivity(sim, sk, 1e-2, 1)
rel = float(cp.linalg.norm(grad_pml - grad_sup) / cp.linalg.norm(grad_sup))
print(f"[check 1] PML vs superposition gradient relative L2 {rel:.3e}  (expect << 1e-3)")

# ---------------------------------------------------------------------------
# check 2: directional FD gradient check, convergence under refinement
print("[check 2] directional gradient relerr (contrast 1.5, should shrink as h->0):")
for Nx, N in [((40, 40), 120), ((76, 76), 240), ((112, 112), 360)]:
    rel, an, fd = directional_relerr(Nx, N, 1.5)
    print(f"   {Nx[0]:3d}^2  N={N:3d}   analytic {an:+.4e}  FD {fd:+.4e}   relerr {rel:.3e}")

# ---------------------------------------------------------------------------
# check 3: sponge absorption. fraction of field energy remaining at the final step
from solvers.wave import (build_materials, compile_kernels, define_step_method,
                          define_homogeneous_Neumann_BC, define_excitation)
sim, source, sensors, um, (RHO1, KAPPA1) = build_case((40, 40), 120, 2643.0 / 1.204)


def remaining_energy(damping):
    air = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    mat = build_materials(sim, air, damping)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    exc = define_excitation(sim, source.position, kernels, mat)
    u0 = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    u1 = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    for t in range(sim.N):
        u0 = fd_step(u0, u1, u0)
        u0 = exc(u0, source.signal, t)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
    return float(cp.sum(u1 ** 2))


for width, mult in [(8, 0.6), (12, 2.0), (16, 6.0)]:
    sponge = build_sponge(sim, width=width, d_max=mult / (KAPPA1 * sim.dt), power=3)
    frac = remaining_energy(sponge) / remaining_energy(zero)
    print(f"[check 3] width={width:2d} d_max_mult={mult:4.1f}  retained fraction {frac:.3e}")
