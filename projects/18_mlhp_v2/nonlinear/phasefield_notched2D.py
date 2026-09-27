import argparse
import time

import cupy as cp
import matplotlib.pyplot as plt
import mlhp
import numpy as np

from helper import QuadratureTables, VoxelMultigrid, isotropic_tensor

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--load", default="tension", choices=["tension", "shear"])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# single edge notched tension or shear (Miehe et al. 2010), units kN and mm
# geometry
LENGTHS = [1.0, 1.0]
NOTCH = 0.5  # notch length from the left edge at mid height, two voxels thick

# discretization
RESOLUTION = args.resolution  # voxels per edge
DEGREE = 1
SUB_VOXELS = 1

# physics
LAMBDA, MU = 121.15, 80.77  # plane strain
GC = 2.7e-3  # fracture toughness
ELL = 0.0075  # phase-field length scale
RESIDUAL = 1e-7  # stiffness left in fully broken material
ALPHA = 1e-6  # stiffness of the notch voxels

# loading
LOAD = args.load
STAGES = {  # (end displacement, increment) of the top face
    "tension": [(0.005, 1e-4), (0.0065, 1e-6)],
    "shear": [(0.02, 2e-5)],
}[LOAD]

# solver
CG_TOL = 1e-6  # relative residual, per staggered iteration
SMOOTHING = (1, 2, 4, 8)  # Chebyshev degree per level, fine to coarse
STAGGER_TOL = 1e-4  # max phase-field change between staggered iterations
MAX_STAGGER = 500
LUMPED = True  # lumped phase-field mass keeps 0 <= d <= 1, needs DEGREE = 1

# ---------------------------------------- setup --------------------------------------
D = 2
nvoxels = [RESOLUTION] * D
cell = [length / n * SUB_VOXELS for length, n in zip(LENGTHS, nvoxels)]
tables = QuadratureTables(mlhp.makeHpTensorSpace, cell, DEGREE, SUB_VOXELS)
volume = tables.volume

# displacement: g K + (1 - g) [tr < 0] K_mean, with the sub-voxel mean dilatation
grad = tables.gradient(D)
b_mean = tables.mean(grad[:, :, 0] + grad[:, :, 3])
KAPPA = LAMBDA + 2.0 * MU / 3.0
K_full = tables.integrate(grad, isotropic_tensor(D, LAMBDA, MU))
K_mean = KAPPA * volume[:, None, None] * b_mean[:, :, None] * b_mean[:, None, :]

# phase field: Gc (M / l + l L) + 2 H M_mean, lumped: g averaged over the nodes
value = tables.value(1)
m_mean = tables.mean(value[:, :, 0])
M = tables.integrate(value)
M_mean = volume[:, None, None] * m_mean[:, :, None] * m_mean[:, None, :]
if LUMPED:
    M = M.sum(axis=2)[:, :, None] * np.eye(M.shape[1])
    M_mean = M_mean.sum(axis=2)[:, :, None] * np.eye(M.shape[1])
A_crack = M / ELL + ELL * tables.integrate(tables.gradient(1))


def face_dofs(basis, face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


def supports(basis):  # bottom and top faces clamped, sides guided vertically in shear
    fixed = [face_dofs(basis, f, i) for f in (2, 3) for i in (0, 1)]
    if LOAD == "shear":
        fixed += [face_dofs(basis, f, 1) for f in (0, 1)]
    return np.unique(np.concatenate(fixed))


tic = time.time()
solver_u = VoxelMultigrid(
    [K_full, K_mean],
    nvoxels,
    LENGTHS,
    DEGREE,
    SUB_VOXELS,
    D,
    supports,
    smoothing=SMOOTHING,
)
solver_d = VoxelMultigrid(
    [A_crack, M_mean],
    nvoxels,
    LENGTHS,
    DEGREE,
    SUB_VOXELS,
    1,
    lambda basis: [],
    smoothing=SMOOTHING,
)
print(f"dofs u {solver_u.ndof} d {solver_d.ndof}, setup {time.time() - tic:.2f} s")

top = cp.asarray(face_dofs(solver_u.basis, 3, 1 if LOAD == "tension" else 0))
x, y = np.meshgrid(
    *[(np.arange(n) + 0.5) / n * L for n, L in zip(nvoxels, LENGTHS)], indexing="ij"
)
notch = (x < NOTCH) & (np.abs(y - 0.5 * LENGTHS[1]) < LENGTHS[1] / RESOLUTION)
stiffness = cp.asarray(np.where(notch, ALPHA, 1.0))

# ------------------------------------- simulation ------------------------------------
loads = np.concatenate(
    [
        np.arange(start, end, inc) + inc
        for (start, _), (end, inc) in zip([(0.0, 0)] + STAGES, STAGES)
    ]
)
u = cp.zeros(solver_u.ndof)
w = None
d = cp.zeros(solver_d.ndof)
history = cp.zeros(nvoxels)
reactions, staggers, cg_u, cg_d = [], [], [], []

tic = time.time()
for step, load in enumerate(loads):
    u_dirichlet = cp.zeros(solver_u.ndof)
    u_dirichlet[top] = load
    w = u * (load / loads[step - 1]) - u_dirichlet if step else None  # linear predictor
    for stagger in range(MAX_STAGGER):
        if LUMPED:
            g = (
                solver_d.voxel_values((1.0 - cp.clip(d, 0.0, 1.0)) ** 2, m_mean)
                + RESIDUAL
            )
        else:
            g = (
                1.0 - cp.clip(solver_d.voxel_values(d, m_mean), 0.0, 1.0)
            ) ** 2 + RESIDUAL
        compressed = solver_u.voxel_values(u, b_mean) < 0.0
        for switch in range(20):  # semismooth Newton on the tension/compression split
            solver_u.update([stiffness * g, stiffness * (1.0 - g) * compressed], False)
            lift = -solver_u.apply_full(u_dirichlet)
            w, it = solver_u.solve(lift, x0=w, rtol=CG_TOL)
            cg_u.append(it)
            u = u_dirichlet + w
            new = solver_u.voxel_values(u, b_mean) < 0.0
            if bool((new == compressed).all()):
                break
            compressed = new
        energy_full, energy_mean = solver_u.element_energy(u)
        psi_plus = (
            0.5 * stiffness * (energy_full - compressed * energy_mean) / volume[0]
        )
        driving = cp.maximum(history, psi_plus)
        solver_d.update([cp.full(nvoxels, GC), 2.0 * driving], False)
        rhs = solver_d.load(2.0 * driving, volume[:, None] * m_mean)
        d_new, it = solver_d.solve(rhs, x0=d, rtol=CG_TOL)
        cg_d.append(it)
        change = float(cp.abs(d_new - d).max())
        d = d_new
        if change < STAGGER_TOL:
            break
    history = driving
    reactions.append(float(solver_u.apply_full(u)[top].sum()))
    staggers.append(stagger + 1)
    if step % 50 == 0 or stagger > 5:
        print(
            f"step {step:5d} u {load:.2e} reaction {reactions[-1]:.4f} "
            f"staggers {stagger + 1:3d} max d {float(d.max()):.3f} "
            f"elapsed {time.time() - tic:.1f} s",
            flush=True,
        )
    if max(reactions) > 0.1 and reactions[-1] < 0.02 * max(reactions):
        break
toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {step + 1} steps, {sum(staggers)} staggered "
    f"iterations, cg iterations u mean {np.mean(cg_u):.1f} max {max(cg_u)} "
    f"d mean {np.mean(cg_d):.1f} max {max(cg_d)}, peak reaction {max(reactions):.4f}"
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
d_voxels = solver_d.voxel_values(d, m_mean).get()
ax.imshow(
    np.ma.masked_where(notch, d_voxels).T,
    origin="lower",
    cmap="turbo",
    vmin=0.0,
    vmax=1.0,
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
ax.plot(loads[: len(reactions)], reactions, "k")
ax.set_xlabel("displacement [mm]")
ax.set_ylabel("reaction force [kN]")
plt.show()
