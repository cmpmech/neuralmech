import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32
THREADS_J, THREADS_I = 128, 4  # THREADS_J = TILE_J in temporal_step
TILE_I = 48  # as in temporal_step
STEPS_PER_LAUNCH = 8  # as STEPS in temporal_step
COMPILER_OPTIONS = ("--use_fast_math",)

# physics
LENGTHS = [1.0, 2.0]
WAVESPEED = 1.0
T = 1.0

# discretization
NX, NY = 8000, 16000  # beyond the L2 cache, where temporal blocking pays off
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# ------------------------------------ prepare data -----------------------------------
dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)
# an even number of launches, so the two pairs of arrays return to their roles
N = math.ceil(T / dt / (2 * STEPS_PER_LAUNCH)) * 2 * STEPS_PER_LAUNCH

# the row stride is rounded up to a warp, so each row starts on an aligned address
NX_PADDED = NX
NY_PADDED = ((NY + 32 - 1) // 32) * 32

U = cp.zeros((2, NX_PADDED, NY_PADDED), dtype=DTYPE)
V = cp.zeros_like(U)  # the tiles overlap, so the new levels go to a second pair

# the gaussian is separable, so no full-size temporaries are needed
x = cp.linspace(0, LENGTHS[0], NX, dtype=DTYPE)[:, None]
y = cp.linspace(0, LENGTHS[1], NY, dtype=DTYPE)[None, :]
gaussian_x = cp.exp(-((x - CENTER[0]) ** 2) / (2 * SIGMA**2))
gaussian_y = cp.exp(-((y - CENTER[1]) ** 2) / (2 * SIGMA**2))
cp.multiply(gaussian_x, gaussian_y, out=U[0, :NX, :NY])
U[1] = U[0]

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR / "step2D_wave.cu").read_text()
module = cp.RawModule(code=source, options=COMPILER_OPTIONS)
temporal_kernel = module.get_function("temporal_step")
padded_kernel = module.get_function("padded_step")

# each block updates its tile without the halo of STEPS_PER_LAUNCH points per side
inner_j = THREADS_J - 2 * STEPS_PER_LAUNCH
inner_i = TILE_I - 2 * STEPS_PER_LAUNCH
blocks_j = (NY + inner_j - 1) // inner_j
blocks_i = (NX + inner_i - 1) // inner_i
padded_blocks_j = (NY_PADDED + THREADS_J - 1) // THREADS_J
padded_blocks_i = (NX_PADDED + THREADS_I - 1) // THREADS_I


def fd_steps(u0, u1, v0, v1):
    temporal_kernel(
        (blocks_j, blocks_i),
        (THREADS_J, THREADS_I),
        (
            u0, u1, v0, v1,
            DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy),
            NX, NY, NY_PADDED,
        ),
    )


def fd_step(u0, u1, u2):
    padded_kernel(
        (padded_blocks_j, padded_blocks_i),
        (THREADS_J, THREADS_I),
        (
            u0, u1, u2,
            DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy),
            NX, NY, NY_PADDED,
        ),
    )
    return u2


# --------------------------------------- solve ---------------------------------------
w0, w1 = U.copy()  # the same steps with padded_step, for comparison
fd_step(w0, w1, cp.empty_like(w0))  # warm-up: the first launch compiles the kernel
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in range(N):
    w0 = fd_step(w0, w1, w0)
    w1, w0 = w0, w1
cp.cuda.Stream.null.synchronize()
padded = (time.time() - tic) / N

u0, u1 = U
v0, v1 = V
fd_steps(u0, u1, v0, v1)  # warm-up, writes only v0 and v1
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in range(N // STEPS_PER_LAUNCH):
    fd_steps(u0, u1, v0, v1)
    u0, u1, v0, v1 = v0, v1, u0, u1
cp.cuda.Stream.null.synchronize()
toc = time.time()

dofs = (NX - 2) * (NY - 2)
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / N) / 1e9:.2f} billion dofs/s")
print(f"padded_step {padded * 1e3:.4f} ms/step ({dofs / padded / 1e9:.2f} billion dofs/s)")
print(f"largest difference to padded_step {float(cp.abs(u1 - w1).max()):.2e}")

# ----------------------------------- postprocessing ----------------------------------
scale = 0.1
# every fifth point, the full grid is too large to draw
x = np.linspace(0, LENGTHS[0], NX)[::5]
y = np.linspace(0, LENGTHS[1], NY)[::5]
x, y = np.meshgrid(x, y, indexing="ij")
fig, ax = plt.subplots(figsize=(3, 6), dpi=100)
ax.pcolormesh(x, y, u1[:NX:5, :NY:5].get(), cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
