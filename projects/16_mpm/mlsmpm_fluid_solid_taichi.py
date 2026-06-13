import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

np.random.seed(0)
try:
    ti.init(arch=ti.gpu)  # cuda/vulkan/metal where available
except Exception:
    ti.init(arch=ti.cpu)  # fall back when no usable GPU driver is present

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
rho = 1.0
FLUID, SOLID = 0, 1
E_s = 1e2
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

snapshot_steps = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
anim_every = 20

# ----------------------------- particle seeding -------------------------
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

x_np = np.concatenate(positions, axis=0).astype(np.float32)
material_np = np.concatenate(materials, axis=0).astype(np.int32)
n_p = x_np.shape[0]
fluid = material_np == FLUID
mu_np = np.where(material_np == SOLID, mu_s, 0.0).astype(np.float32)
lam_np = np.where(material_np == SOLID, lam_s, lam_f).astype(np.float32)

# ------------------------------- ti fields ------------------------------
x = ti.Vector.field(2, float, n_p)
v = ti.Vector.field(2, float, n_p)
C = ti.Matrix.field(2, 2, float, n_p)
F = ti.Matrix.field(2, 2, float, n_p)
material = ti.field(int, n_p)
mu_p = ti.field(float, n_p)
lam_p = ti.field(float, n_p)
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


@ti.kernel
def reset():
    for p in x:
        v[p] = [0.0, 0.0]
        C[p] = ti.Matrix.zero(float, 2, 2)
        F[p] = ti.Matrix.identity(float, 2)


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

    for p in x:  # particle to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        # advance the deformation gradient, keep the fluid volume-only
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        J = F[p].determinant()
        if material[p] == FLUID:
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)

        eye = ti.Matrix.identity(float, 2)
        tau = mu_p[p] * (F[p] @ F[p].transpose() - eye) + lam_p[p] * ti.log(J) * eye
        stress = -(dt * p_vol * 4.0 * inv_dx * inv_dx) * tau
        affine = stress + p_mass * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:  # grid update: gravity + wall boundary conditions
        if grid_m[i, j] > 0.0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j][1] -= dt * gravity
            if i < bound and grid_v[i, j][0] < 0.0:
                grid_v[i, j][0] = 0.0
            if i > n_grid - bound and grid_v[i, j][0] > 0.0:
                grid_v[i, j][0] = 0.0
            if j < bound and grid_v[i, j][1] < 0.0:
                grid_v[i, j][1] = 0.0
            if j > n_grid - bound and grid_v[i, j][1] > 0.0:
                grid_v[i, j][1] = 0.0

    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            g_v = grid_v[base + offset]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4.0 * inv_dx * inv_dx * weight * g_v.outer_product(dpos)
        v[p] = new_v
        C[p] = new_C
        x[p] = ti.math.clamp(x[p] + dt * new_v, bound * dx, 1.0 - bound * dx)


def plot_snapshot(ax, pos, fluid):
    ax.scatter(pos[fluid, 0], pos[fluid, 1], c="steelblue", s=4)
    ax.scatter(pos[~fluid, 0], pos[~fluid, 1], c="crimson", s=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")


# ----------------------------- initialization ---------------------------
x.from_numpy(x_np)
material.from_numpy(material_np)
mu_p.from_numpy(mu_np)
lam_p.from_numpy(lam_np)
reset()

# --------------------------------- solve --------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

snapshots = []
tic = time.time()
for s in range(n_steps):
    if s in snapshot_steps:
        snapshots.append((s, x.to_numpy()))
    if args.animate and s % anim_every == 0:
        fig, ax = plt.subplots()
        plot_snapshot(ax, x.to_numpy(), fluid)
        fig.savefig(ANIMATION_DIR / f"mpm_{s // anim_every:04d}.png", dpi=120)
        plt.close(fig)
    substep()
ti.sync()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"particles {n_p}, steps {n_steps}, ms/step {(toc - tic) / n_steps * 1e3:.2f}")

# ----------------------------- postprocessing ---------------------------
if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for s, pos in snapshots:
        fig, ax = plt.subplots()
        plot_snapshot(ax, pos, fluid)
        fig.savefig(RESULTS_DIR / f"mpm_fluid_solid_taichi_{s:04d}.pdf")
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
