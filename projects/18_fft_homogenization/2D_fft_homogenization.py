import argparse
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from fft_homogenization import FusedHomogenization, fibers, lame
from postprocessing import load_cmap

BASE_DIR = Path(__file__).parent
RGB_PNG_DIR = (BASE_DIR / "../../results/rgb_png").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--n", type=int, default=2048, help="pixels per side")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# physics: stiff fibers in a matrix, plane strain
E_FIBER, E_MATRIX, NU = 20.0, 1.0, 0.3

# solver
RTOL = 1e-4  # relative residual, the effective stiffness is then accurate to about 1e-5

# --------------------------------------- setup ---------------------------------------
phase = fibers(args.n)
lam, mu = lame(np.where(phase > 0, E_FIBER, E_MATRIX), NU)
homogenization = FusedHomogenization(lam, mu)
homogenization.cg(RTOL)  # compile the kernels and plan the transforms

# --------------------------------------- solve ---------------------------------------
cp.cuda.Device().synchronize()
tic = time.time()
eps, iterations = homogenization.cg(RTOL)
C = homogenization.effective(eps).get()
print(f"{args.n}^2 pixels, 3 load cases: {time.time() - tic:.3f} s, {iterations} iterations")
print(f"effective stiffness (Voigt, fiber volume fraction {phase.mean():.3f}):\n{C}")

# ----------------------------------- postprocessing ----------------------------------
strain = eps[0, 0].get()  # eps_xx of the load case eps_xx = 1
fig, ax = plt.subplots(figsize=(args.n / 300, args.n / 300), dpi=300)
ax.imshow(strain.T, origin="lower", cmap=load_cmap(CMAP_DIR / "rainbow_desaturated.cmap"),
          vmin=np.percentile(strain, 1), vmax=np.percentile(strain, 99))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
else:
    plt.savefig(RGB_PNG_DIR / "fft_homogenization.png")
