import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

np.random.seed(0)
wp.init()
device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"

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

# compile-time constants visible to the kernels
N_GRID = wp.constant(n_grid)
DX = wp.constant(dx)
INV_DX = wp.constant(inv_dx)
DT = wp.constant(dt)
GRAVITY = wp.constant(gravity)
BOUND = wp.constant(bound)
P_VOL = wp.constant(p_vol)
P_MASS = wp.constant(p_mass)

# ------------------------------- kernels --------------------------------
EYE = wp.constant(wp.mat22(1.0, 0.0, 0.0, 1.0))


@wp.func
def bspline(fx: float):
    # quadratic B-spline weights for the 3-node stencil along one axis
    a = 1.5 - fx
    b = fx - 1.0
    c = fx - 0.5
    return wp.vec3(0.5 * a * a, 0.75 - b * b, 0.5 * c * c)


@wp.kernel
def clear_grid(grid_v: wp.array(dtype=wp.vec2), grid_m: wp.array(dtype=float)):
    i = wp.tid()
    grid_v[i] = wp.vec2(0.0, 0.0)
    grid_m[i] = 0.0


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    F: wp.array(dtype=wp.mat22),
    C: wp.array(dtype=wp.mat22),
    fluid: wp.array(dtype=int),
    mu_p: wp.array(dtype=float),
    lam_p: wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec2),
    grid_m: wp.array(dtype=float),
):
    p = wp.tid()
    Xp = x[p] * INV_DX
    base_x = int(Xp[0] - 0.5)
    base_y = int(Xp[1] - 0.5)
    fx = wp.vec2(Xp[0] - float(base_x), Xp[1] - float(base_y))
    wx = bspline(fx[0])
    wy = bspline(fx[1])

    # advance the deformation gradient, keep the fluid volume-only
    Fp = (EYE + DT * C[p]) * F[p]
    J = wp.determinant(Fp)
    if fluid[p] == 1:
        s = wp.sqrt(J)
        Fp = wp.mat22(s, 0.0, 0.0, s)
    F[p] = Fp

    # unified Kirchhoff stress folded into the APIC affine momentum matrix
    tau = mu_p[p] * (Fp * wp.transpose(Fp) - EYE) + lam_p[p] * wp.log(J) * EYE
    stress = -(DT * P_VOL * 4.0 * INV_DX * INV_DX) * tau
    affine = stress + P_MASS * C[p]

    for i in range(3):
        for j in range(3):
            dpos = (wp.vec2(float(i), float(j)) - fx) * DX
            weight = wx[i] * wy[j]
            idx = (base_x + i) * N_GRID + (base_y + j)
            wp.atomic_add(grid_m, idx, weight * P_MASS)
            wp.atomic_add(grid_v, idx, weight * (P_MASS * v[p] + affine * dpos))


@wp.kernel
def grid_op(grid_v: wp.array(dtype=wp.vec2), grid_m: wp.array(dtype=float)):
    idx = wp.tid()
    m = grid_m[idx]
    if m > 0.0:
        gi = idx // N_GRID
        gj = idx - gi * N_GRID
        vx = grid_v[idx][0] / m
        vy = grid_v[idx][1] / m - DT * GRAVITY
        if gi < BOUND and vx < 0.0:
            vx = 0.0
        if gi >= N_GRID - BOUND and vx > 0.0:
            vx = 0.0
        if gj < BOUND and vy < 0.0:
            vy = 0.0
        if gj >= N_GRID - BOUND and vy > 0.0:
            vy = 0.0
        grid_v[idx] = wp.vec2(vx, vy)


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    grid_v: wp.array(dtype=wp.vec2),
):
    p = wp.tid()
    Xp = x[p] * INV_DX
    base_x = int(Xp[0] - 0.5)
    base_y = int(Xp[1] - 0.5)
    fx = wp.vec2(Xp[0] - float(base_x), Xp[1] - float(base_y))
    wx = bspline(fx[0])
    wy = bspline(fx[1])

    new_v = wp.vec2(0.0, 0.0)
    new_C = wp.mat22(0.0, 0.0, 0.0, 0.0)
    for i in range(3):
        for j in range(3):
            dpos = (wp.vec2(float(i), float(j)) - fx) * DX
            weight = wx[i] * wy[j]
            g_v = grid_v[(base_x + i) * N_GRID + (base_y + j)]
            new_v = new_v + weight * g_v
            new_C = new_C + (4.0 * INV_DX * INV_DX * weight) * wp.outer(g_v, dpos)

    v[p] = new_v
    C[p] = new_C
    lo = float(BOUND) * DX
    hi = 1.0 - float(BOUND) * DX
    new_x = x[p] + DT * new_v
    x[p] = wp.vec2(wp.clamp(new_x[0], lo, hi), wp.clamp(new_x[1], lo, hi))


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

x_np = np.concatenate(positions, axis=0).astype(np.float32)
material = np.concatenate(materials, axis=0)
n_p = x_np.shape[0]
fluid = material == FLUID
mu_np = np.where(material == SOLID, mu_s, 0.0).astype(np.float32)
lam_np = np.where(material == SOLID, lam_s, lam_f).astype(np.float32)

x = wp.array(x_np, dtype=wp.vec2, device=device)
v = wp.zeros(n_p, dtype=wp.vec2, device=device)
F = wp.array(
    np.tile(np.eye(2, dtype=np.float32), (n_p, 1, 1)), dtype=wp.mat22, device=device
)
C = wp.zeros(n_p, dtype=wp.mat22, device=device)
fluid_wp = wp.array(fluid.astype(np.int32), dtype=int, device=device)
mu_p = wp.array(mu_np, dtype=float, device=device)
lam_p = wp.array(lam_np, dtype=float, device=device)
grid_v = wp.zeros(n_grid * n_grid, dtype=wp.vec2, device=device)
grid_m = wp.zeros(n_grid * n_grid, dtype=float, device=device)

# --------------------------------- solve --------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

print(f"device {device}, particles {n_p}")
snapshots = []
wp.synchronize()
tic = time.time()
for s in range(n_steps):
    if s in snapshot_steps:
        snapshots.append((s, x.numpy().copy()))
    if args.animate and s % anim_every == 0:
        fig, ax = plt.subplots()
        plot_snapshot(ax, x.numpy(), fluid)
        fig.savefig(ANIMATION_DIR / f"mpm_{s // anim_every:04d}.png", dpi=120)
        plt.close(fig)
    wp.launch(clear_grid, dim=n_grid * n_grid, inputs=[grid_v, grid_m], device=device)
    wp.launch(
        p2g,
        dim=n_p,
        inputs=[x, v, F, C, fluid_wp, mu_p, lam_p, grid_v, grid_m],
        device=device,
    )
    wp.launch(grid_op, dim=n_grid * n_grid, inputs=[grid_v, grid_m], device=device)
    wp.launch(g2p, dim=n_p, inputs=[x, v, C, grid_v], device=device)
wp.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"particles {n_p}, steps {n_steps}, ms/step {(toc - tic) / n_steps * 1e3:.2f}")

# ----------------------------- postprocessing ---------------------------
if args.book:
    for s, pos in snapshots:
        fig, ax = plt.subplots()
        plot_snapshot(ax, pos, fluid)
        fig.savefig(RGB_PDF_DIR / f"mpm_fluid_solid_warp_{s:04d}.pdf")
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
