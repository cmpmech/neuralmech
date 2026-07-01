import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ------------------------------- settings -------------------------------
# background grid (unit square domain, so dx = 1 / n_grid)
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
bound = 3  # wall thickness in cells for the no-penetration boundary

# time stepping
dt = 1e-4
n_steps = 4500
gravity = 50.0

# materials share one Kirchhoff stress tau = mu (F F^T - I) + lam ln(J) I.
# the solid (jelly) carries shear; the fluid sets mu = 0 so tau is a pure
# pressure lam ln(J) I and its F is reset to volume-only every step.
rho = 1.0
FLUID, SOLID = 0, 1

E_s = 1e2  # solid Young's modulus
nu_s = 0.2
mu_s = E_s / (2.0 * (1.0 + nu_s))
lam_s = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))
lam_f = 5e2  # fluid bulk modulus

# particle sampling: particles per cell (a ppc x ppc lattice per cell)
ppc = 2
p_vol = (dx * dx) / (ppc * ppc)
p_mass = p_vol * rho

# regions: x0, x1, y0, y1, material
regions = [
    (0.12, 0.88, 0.05, 0.20, FLUID),  # pool of fluid on the floor
    (0.42, 0.57, 0.55, 0.70, SOLID),  # jelly block dropped into it
]

# snapshots captured for the figure (and frame stride for --animate)
snapshot_steps = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
anim_every = 20

# ------------------------------- helpers --------------------------------
eye2 = np.eye(2)


def bspline_weights(fx):
    # quadratic B-spline weights for the 3-node stencil, per dimension
    # fx is the particle offset from its base node, in cell units, in [0.5, 1.5]
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return np.stack([w0, w1, w2], axis=1)  # (n_p, 3)


def step(x, v, F, C, fluid, mu_p, lam_p):
    # advance the deformation gradient with the previous affine velocity
    F = np.einsum("pab,pbc->pac", eye2[None] + dt * C, F)
    J = np.linalg.det(F)
    F[fluid] = np.sqrt(J[fluid])[:, None, None] * eye2[None]  # keep fluid volume-only

    # unified Kirchhoff stress tau = mu (F F^T - I) + lam ln(J) I
    FFt = np.einsum("pab,pcb->pac", F, F)
    tau = (mu_p[:, None, None] * (FFt - eye2[None])
           + lam_p[:, None, None] * np.log(J)[:, None, None] * eye2[None])
    stress = -(dt * p_vol * 4.0 * inv_dx * inv_dx) * tau
    affine = stress + p_mass * C  # APIC affine momentum matrix

    grid_v = np.zeros((n_grid * n_grid, 2))
    grid_m = np.zeros(n_grid * n_grid)

    base = (x * inv_dx - 0.5).astype(int)  # (n_p, 2)
    fx = x * inv_dx - base  # (n_p, 2), in [0.5, 1.5]
    wx = bspline_weights(fx[:, 0])  # (n_p, 3)
    wy = bspline_weights(fx[:, 1])

    # ----- particle to grid (P2G): scatter mass and momentum over 3x3 stencil
    for i in range(3):
        for j in range(3):
            weight = wx[:, i] * wy[:, j]  # (n_p,)
            node = (base[:, 0] + i) * n_grid + (base[:, 1] + j)
            dpos = (np.array([i, j]) - fx) * dx  # node minus particle, physical
            momentum = p_mass * v + np.einsum("pab,pb->pa", affine, dpos)
            np.add.at(grid_m, node, weight * p_mass)
            np.add.at(grid_v, node, weight[:, None] * momentum)

    # ----- grid update: momentum -> velocity, gravity, wall boundary conditions
    grid_v = grid_v.reshape(n_grid, n_grid, 2)
    grid_m = grid_m.reshape(n_grid, n_grid)
    nonempty = grid_m > 0
    grid_v[nonempty] /= grid_m[nonempty][:, None]
    grid_v[..., 1] -= dt * gravity

    grid_v[:bound, :, 0] = np.maximum(grid_v[:bound, :, 0], 0.0)
    grid_v[-bound:, :, 0] = np.minimum(grid_v[-bound:, :, 0], 0.0)
    grid_v[:, :bound, 1] = np.maximum(grid_v[:, :bound, 1], 0.0)
    grid_v[:, -bound:, 1] = np.minimum(grid_v[:, -bound:, 1], 0.0)
    grid_v = grid_v.reshape(n_grid * n_grid, 2)

    # ----- grid to particle (G2P): gather velocity, rebuild affine, advect
    new_v = np.zeros_like(v)
    new_C = np.zeros_like(C)
    for i in range(3):
        for j in range(3):
            weight = wx[:, i] * wy[:, j]
            node = (base[:, 0] + i) * n_grid + (base[:, 1] + j)
            g_v = grid_v[node]  # (n_p, 2)
            dpos = (np.array([i, j]) - fx) * dx
            new_v += weight[:, None] * g_v
            new_C += 4.0 * inv_dx * inv_dx * weight[:, None, None] * np.einsum(
                "pa,pb->pab", g_v, dpos)

    x = x + dt * new_v
    x = np.clip(x, bound * dx, 1.0 - bound * dx)  # safety: keep particles in-domain
    return x, new_v, F, new_C


def plot_snapshot(ax, pos, fluid):
    ax.scatter(pos[fluid, 0], pos[fluid, 1], c="steelblue", s=4)
    ax.scatter(pos[~fluid, 0], pos[~fluid, 1], c="crimson", s=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")


# ----------------------------- initialization ---------------------------
positions = []
materials = []
for x0, x1, y0, y1, mat in regions:
    nx = round((x1 - x0) / dx * ppc)
    ny = round((y1 - y0) / dx * ppc)
    gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    gy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    mesh = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    positions.append(mesh)
    materials.append(np.full(mesh.shape[0], mat))

x = np.concatenate(positions, axis=0)
material = np.concatenate(materials, axis=0)
n_p = x.shape[0]
fluid = material == FLUID
mu_p = np.where(material == SOLID, mu_s, 0.0)
lam_p = np.where(material == SOLID, lam_s, lam_f)

v = np.zeros((n_p, 2))
F = np.broadcast_to(eye2, (n_p, 2, 2)).copy()
C = np.zeros((n_p, 2, 2))

# --------------------------------- solve --------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

snapshots = []
tic = time.time()
for s in range(n_steps):
    if s in snapshot_steps:
        snapshots.append((s, x.copy()))
    if args.animate and s % anim_every == 0:
        fig, ax = plt.subplots()
        plot_snapshot(ax, x, fluid)
        fig.savefig(ANIMATION_DIR / f"mpm_{s // anim_every:04d}.png", dpi=120)
        plt.close(fig)
    x, v, F, C = step(x, v, F, C, fluid, mu_p, lam_p)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"particles {n_p}, steps {n_steps}, ms/step {(toc - tic) / n_steps * 1e3:.2f}")

# ----------------------------- postprocessing ---------------------------
if args.book:
    for s, pos in snapshots:
        fig, ax = plt.subplots()
        plot_snapshot(ax, pos, fluid)
        fig.savefig(RGB_PDF_DIR / f"mpm_fluid_solid_{s:04d}.pdf")
        plt.close(fig)
elif args.animate:
    pass
else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (s, pos) in zip(axes.ravel(), snapshots):
        plot_snapshot(ax, pos, fluid)
        ax.set_title(f"step {s}")
    fig.tight_layout()
    plt.show()
