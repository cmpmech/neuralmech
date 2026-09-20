import argparse
import math
import time
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

wp.init()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = float  # wp.float64 for double precision

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

x = np.linspace(0, LENGTHS[0], NX)
y = np.linspace(0, LENGTHS[1], NY)
x, y = np.meshgrid(x, y, indexing="ij")
gaussian = np.exp(-((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / (2 * SIGMA**2))
u_init = gaussian.astype(np.float32)

u0 = wp.array(u_init, dtype=DTYPE, device="cuda")
u1 = wp.array(u_init, dtype=DTYPE, device="cuda")


# --------------------------------------- setup ---------------------------------------
@wp.kernel(enable_backward=False)
def wave_step_kernel(
    u_prev: wp.array2d(dtype=DTYPE),
    u_curr: wp.array2d(dtype=DTYPE),
    u_next: wp.array2d(dtype=DTYPE),  # aliases u_prev, which is overwritten in place
    c_sq: float,
    dt_sq: float,
    inv_dx_sq: float,
    inv_dy_sq: float,
):
    i, j = wp.tid()
    nx, ny = u_curr.shape[0], u_curr.shape[1]

    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
        center = u_curr[i, j]
        laplacian_x = (u_curr[i - 1, j] - 2.0 * center + u_curr[i + 1, j]) * inv_dx_sq
        laplacian_y = (u_curr[i, j - 1] - 2.0 * center + u_curr[i, j + 1]) * inv_dy_sq
        u_next[i, j] = (
            2.0 * center
            - u_prev[i, j]
            + (c_sq * dt_sq) * (laplacian_x + laplacian_y)
        )


c_sq = WAVESPEED**2
dt_sq = dt**2
inv_dx_sq = 1.0 / dx**2
inv_dy_sq = 1.0 / dy**2

# --------------------------------------- solve ---------------------------------------
wp.synchronize()
tic = time.time()
for t in tqdm(range(N)):
    wp.launch(
        kernel=wave_step_kernel,
        dim=(NX, NY),
        inputs=[u0, u1, u0, c_sq, dt_sq, inv_dx_sq, inv_dy_sq],
        device="cuda",
        block_dim=128,
    )
    u1, u0 = u0, u1
wp.synchronize()
toc = time.time()

dofs = (NX - 2) * (NY - 2)
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / N) / 1e9:.2f} billion dofs/s")

# ----------------------------------- postprocessing ----------------------------------
scale = 0.1
fig, ax = plt.subplots(figsize=(3, 6), dpi=100)
ax.pcolormesh(x, y, u1.numpy(), cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / "wave_warp.pdf")
