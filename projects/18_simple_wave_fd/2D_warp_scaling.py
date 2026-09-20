import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from tqdm import tqdm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

wp.init()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = float  # wp.float64 for double precision

# physics
LENGTHS = [1.0, 2.0]
WAVESPEED = 1.0

# discretization
RESOLUTIONS = np.logspace(0.5, 4.5, 50).astype(np.int32)
STEPS = 2000
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02


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


# ---------------------------------- parametric study ---------------------------------
dofs = []
timings = []
for resolution in RESOLUTIONS:
    NX = NY = int(resolution)
    dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
    dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)

    x = np.linspace(0, LENGTHS[0], NX)
    y = np.linspace(0, LENGTHS[1], NY)
    x, y = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / (2 * SIGMA**2))
    u_init = gaussian.astype(np.float32)

    u0 = wp.array(u_init, dtype=DTYPE, device="cuda")
    u1 = wp.array(u_init, dtype=DTYPE, device="cuda")

    c_sq = WAVESPEED**2
    dt_sq = dt**2
    inv_dx_sq = 1.0 / dx**2
    inv_dy_sq = 1.0 / dy**2

    wp.synchronize()
    tic = time.time()
    for t in tqdm(range(STEPS), desc=f"warp {NX}", ncols=90):
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

    timings.append((toc - tic) / STEPS)
    dofs.append((NX - 2) * (NY - 2))

# --------------------------------------- export --------------------------------------
save_csv(CSV_DIR / "warp_scaling.csv", x=dofs, y=timings)

# ----------------------------------- postprocessing ----------------------------------
throughput = np.array(dofs) / np.array(timings)
print(f"peak {throughput.max() / 1e9:.2f} billion dofs/s")
print(f"largest grid {max(dofs) / 1e6:.1f} million dofs")

fig, ax = plt.subplots()
ax.plot(dofs, throughput, "k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("dofs")
ax.set_ylabel("dofs/s")
plt.show()
