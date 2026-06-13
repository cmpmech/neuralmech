import argparse
import time
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ------------------------------- settings -------------------------------
# domain (lattice nodes) and obstacle
nx, ny = 360, 120
cx, cy = nx // 4, ny // 2
r = ny // 9  # cylinder radius

# flow: Reynolds number sets the relaxation rate omega via the viscosity
Re = 160.0
u_lb = 0.04  # inlet speed in lattice units (keep << 1 for low Mach)
nu = u_lb * (2 * r) / Re
omega = 1.0 / (3.0 * nu + 0.5)  # BGK relaxation, tau = 1/omega

max_iter = 16000
anim_every = 100

# D2Q9 lattice velocities, weights, and opposite directions for bounce-back
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
              1 / 36, 1 / 36, 1 / 36, 1 / 36])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
left = np.arange(9)[c[:, 0] < 0]    # populations leaving through the right wall
center = np.arange(9)[c[:, 0] == 0]
right = np.arange(9)[c[:, 0] > 0]   # unknown incoming populations at the inlet

# ------------------------------- helpers --------------------------------
def equilibrium(rho, u):
    cu = 3.0 * (c[:, 0][:, None, None] * u[0] + c[:, 1][:, None, None] * u[1])
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho[None] * w[:, None, None] * (1.0 + cu + 0.5 * cu ** 2 - usqr)


def macroscopic(f):
    rho = f.sum(axis=0)
    u = np.einsum("ad,axy->dxy", c, f) / rho
    return rho, u


# obstacle mask and the slightly perturbed inlet velocity profile
xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
obstacle = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
vel = np.zeros((2, nx, ny))
vel[0] = u_lb * (1.0 + 1e-4 * np.sin(yy / (ny - 1) * 2.0 * np.pi))


def plot_vorticity(ax, u):
    vort = np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)
    vort = np.ma.masked_where(obstacle, vort)
    scale = 0.6 * np.abs(vort).max()
    ax.imshow(vort.T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale)
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------- initialization ---------------------------
f = equilibrium(np.ones((nx, ny)), vel)

# --------------------------------- solve --------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

tic = time.time()
for it in range(max_iter):
    # outflow: copy left-going populations from the last interior column
    f[left, -1, :] = f[left, -2, :]

    rho, u = macroscopic(f)

    # inlet (Zou/He): impose velocity, infer density, set incoming populations
    u[:, 0, :] = vel[:, 0, :]
    rho[0, :] = (f[center, 0, :].sum(axis=0)
                 + 2.0 * f[left, 0, :].sum(axis=0)) / (1.0 - u[0, 0, :])
    feq = equilibrium(rho, u)
    f[right, 0, :] = feq[right, 0, :] + f[opp[right], 0, :] - feq[opp[right], 0, :]

    # BGK collision
    fout = f - omega * (f - feq)

    # bounce-back on the cylinder (no-slip)
    fout[:, obstacle] = f[opp][:, obstacle]

    # streaming
    for i in range(9):
        f[i] = np.roll(np.roll(fout[i], c[i, 0], axis=0), c[i, 1], axis=1)

    if args.animate and it % anim_every == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_vorticity(ax, u)
        fig.savefig(ANIMATION_DIR / f"lbm_{it // anim_every:04d}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"grid {nx}x{ny}, iters {max_iter}, ms/iter {(toc - tic) / max_iter * 1e3:.2f}")

# ----------------------------- postprocessing ---------------------------
_, u = macroscopic(f)

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2))
    plot_vorticity(ax, u)
    fig.savefig(RESULTS_DIR / "lbm_d2q9_cylinder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
elif args.animate:
    pass
else:
    fig, ax = plt.subplots(figsize=(6, 2))
    plot_vorticity(ax, u)
    plt.show()
