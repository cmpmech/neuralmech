import argparse
import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32

# physics
LENGTHS = [1.0, 2.0]
WAVESPEED = 1.0
T = 1.0

# discretization
NX, NY = 1600, 3200
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# ------------------------------------ prepare data -----------------------------------
dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)
N = math.ceil(T / dt)

U = cp.zeros((2, NX, NY), dtype=DTYPE)
u0 = U[0]
u1 = U[1]

x = np.linspace(0, LENGTHS[0], NX)
y = np.linspace(0, LENGTHS[1], NY)
x, y = np.meshgrid(x, y, indexing="ij")
gaussian = np.exp(-((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / (2 * SIGMA**2))
u0[:] = cp.asarray(gaussian)
u1[:] = u0[:]


# --------------------------------------- setup ---------------------------------------
def fd_step(u0, u1, u2):
    laplacian_x = (u1[:-2, 1:-1] - 2 * u1[1:-1, 1:-1] + u1[2:, 1:-1]) / dx**2
    laplacian_y = (u1[1:-1, :-2] - 2 * u1[1:-1, 1:-1] + u1[1:-1, 2:]) / dy**2
    u2[1:-1, 1:-1] = (
        -u0[1:-1, 1:-1]
        + 2 * u1[1:-1, 1:-1]
        + dt**2 * WAVESPEED**2 * (laplacian_x + laplacian_y)
    )
    return u2


# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in tqdm(range(N)):
    u0 = fd_step(u0, u1, u0)
    u1, u0 = u0, u1
cp.cuda.Stream.null.synchronize()
toc = time.time()

dofs = (NX - 2) * (NY - 2)
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / N) / 1e9:.2f} billion dofs/s")

# ----------------------------------- postprocessing ----------------------------------
scale = 0.1
fig, ax = plt.subplots(figsize=(3, 6), dpi=100)
ax.pcolormesh(x, y, u1.get(), cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / "wave_cupy.pdf")
