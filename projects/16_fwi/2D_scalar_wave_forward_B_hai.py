import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from numeric import sineburst

from solvers.wave import scalar_simulation, setup_source, simulate

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
THREADS = (4, 128)

# physics
WAVESPEED = 6000.0
DENSITY = 2700.0
AMPLITUDE = 1e8
CYCLES = 1
T = 8e-5
MIN_INDICATOR = 1e-3

# --------------------------------------- setup ---------------------------------------
# load material from CT scan
indicator = np.ascontiguousarray(np.load(DATA_DIR / "B_Hai_1.npy").T)
indicator[:50, :50] = 1.0
indicator[:5, :] = 1.0
indicator[-6:, :] = 1.0
indicator[:, :5] = 1.0
indicator[:, -6:] = 1.0
Nx, Ny = indicator.shape
indicator[indicator == 0] = MIN_INDICATOR

Lx, Ly = 0.04, 0.04 * Ny / Nx
dx = (Lx / (Nx - 3), Ly / (Ny - 3))
dt = 0.95 * min(dx) / WAVESPEED / math.sqrt(2)
frequency = WAVESPEED / 10.0 / dx[0]
N = math.ceil(T / dt)

sim = scalar_simulation(
    (Nx, Ny), dx, N, dt, THREADS, wavespeed=WAVESPEED, density=DENSITY
)

# source
x = np.linspace(-dx[0], Lx + dx[0], Nx)
y = np.linspace(-dx[1], Ly + dx[1], Ny)
x, y = np.meshgrid(x, y, indexing="ij")

source_pos = np.vstack(np.where(np.isclose(x, 0.0) & np.isclose(y, 0.0)))
source_pos = cp.asarray(source_pos, dtype=cp.int32)

t = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t, AMPLITUDE, frequency, CYCLES) / np.prod(dx)  # for the dirac
signal = cp.asarray(
    np.tile(signal_np[:, None], (1, source_pos.shape[1])), dtype=sim.dtype
)
source = setup_source(source_pos, signal)

indicator_padded = cp.ones(sim.Nx_padded, dtype=sim.dtype)
indicator_padded[:Nx, :Ny] = cp.asarray(indicator, dtype=sim.dtype)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator_padded)
cp.cuda.Stream.null.synchronize()
toc = time.time()
dofs = (Nx - 2) * (Ny - 2)
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / N) / 1e9:.2f} billion dofs/s")

# ----------------------------------- postprocessing ----------------------------------
u_np = u.get()
u_masked = np.ma.masked_where(indicator == MIN_INDICATOR, u_np)
indicator_masked = np.ma.masked_where(indicator != MIN_INDICATOR, indicator)
scale = float(np.max(np.abs(u_np))) * 3e-2

fig, ax = plt.subplots(figsize=(Nx / 150, Ny / 150), dpi=150)
ax.pcolormesh(x, y, u_masked, cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.pcolormesh(x, y, indicator_masked * 0 + 0.7, cmap="Greys_r", vmin=0, vmax=1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RGB_PDF_DIR / "CTwaves2D.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
