import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuwave.wave import simulate

from helper import MIN_INDICATOR, Lx, Ly, Nx, Ny, build, dx, indicator

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

# -------------------------------------- settings -------------------------------------
T = 8e-5

# --------------------------------------- setup ---------------------------------------
sim, source, indicator_padded = build(T)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator_padded)
cp.cuda.Stream.null.synchronize()
toc = time.time()
dofs = (Nx - 2) * (Ny - 2)
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / sim.N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / sim.N) / 1e9:.2f} billion dofs/s")

# ----------------------------------- postprocessing ----------------------------------
x = np.linspace(-dx[0], Lx + dx[0], Nx)
y = np.linspace(-dx[1], Ly + dx[1], Ny)
x, y = np.meshgrid(x, y, indexing="ij")

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
