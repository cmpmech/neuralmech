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
n_steps = 4000
gravity = 50.0

# material (neo-Hookean elastic), Lame parameters from (E, nu)
rho = 1.0
E = 1e2  # 1.0e3
nu = 0.2
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# particle sampling: particles per cell (a ppc x ppc lattice per cell)
ppc = 2
p_vol = (dx * dx) / (ppc * ppc)
p_mass = p_vol * rho

# two elastic blocks, stacked so the upper one falls onto the lower one
blocks = [
    (0.40, 0.55, 0.62, 0.77),  # x0, x1, y0, y1  (upper block)
    (0.42, 0.57, 0.30, 0.45),  # lower block
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


def neo_hookean_pk1(F):
    # first Piola-Kirchhoff stress P = mu (F - F^-T) + lam ln(J) F^-T
    J = np.linalg.det(F)
    F_invT = np.swapaxes(np.linalg.inv(F), 1, 2)
    return mu * (F - F_invT) + lam * np.log(J)[:, None, None] * F_invT


def step(x, v, F, C):
    grid_v = np.zeros((n_grid * n_grid, 2))
    grid_m = np.zeros(n_grid * n_grid)

    base = (x * inv_dx - 0.5).astype(int)  # (n_p, 2)
    fx = x * inv_dx - base  # (n_p, 2), in [0.5, 1.5]
    wx = bspline_weights(fx[:, 0])  # (n_p, 3)
    wy = bspline_weights(fx[:, 1])

    # particle stress folded into the APIC affine momentum matrix
    pk1 = neo_hookean_pk1(F)
    stress = -(dt * p_vol * 4.0 * inv_dx * inv_dx) * np.einsum("pij,pkj->pik", pk1, F)
    affine = stress + p_mass * C  # (n_p, 2, 2)

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
            new_C += (
                4.0
                * inv_dx
                * inv_dx
                * weight[:, None, None]
                * np.einsum("pa,pb->pab", g_v, dpos)
            )

    F = np.einsum("pab,pbc->pac", eye2[None] + dt * new_C, F)
    x = x + dt * new_v
    x = np.clip(x, bound * dx, 1.0 - bound * dx)  # safety: keep particles in-domain
    return x, new_v, F, new_C


def plot_snapshot(ax, pos, speed, vmax):
    ax.scatter(pos[:, 0], pos[:, 1], c=speed, cmap="turbo", s=4, vmin=0.0, vmax=vmax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")


# ----------------------------- initialization ---------------------------
positions = []
for x0, x1, y0, y1 in blocks:
    nx = round((x1 - x0) / dx * ppc)
    ny = round((y1 - y0) / dx * ppc)
    gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    gy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    mesh = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    positions.append(mesh)

x = np.concatenate(positions, axis=0)
n_p = x.shape[0]
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
        snapshots.append((s, x.copy(), np.linalg.norm(v, axis=1)))
    if args.animate and s % anim_every == 0:
        fig, ax = plt.subplots()
        plot_snapshot(ax, x, np.linalg.norm(v, axis=1), vmax=8.0)
        fig.savefig(ANIMATION_DIR / f"mpm_{s // anim_every:04d}.png", dpi=120)
        plt.close(fig)
    x, v, F, C = step(x, v, F, C)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"particles {n_p}, steps {n_steps}, ms/step {(toc - tic) / n_steps * 1e3:.2f}")

# ----------------------------- postprocessing ---------------------------
vmax = max(speed.max() for _, _, speed in snapshots)

if args.book:
    for s, pos, speed in snapshots:
        fig, ax = plt.subplots()
        plot_snapshot(ax, pos, speed, vmax)
        fig.savefig(RGB_PDF_DIR / f"mpm_elastic_blocks_{s:04d}.pdf")
        plt.close(fig)
elif args.animate:
    pass
else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (s, pos, speed) in zip(axes.ravel(), snapshots):
        plot_snapshot(ax, pos, speed, vmax)
        ax.set_title(f"step {s}")
    fig.tight_layout()
    plt.show()
