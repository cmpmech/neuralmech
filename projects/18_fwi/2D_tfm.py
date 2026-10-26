import argparse
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cupyx.scipy.signal import hilbert

from plate import CYCLES, FREQUENCY, Lx, Ly, WAVESPEED, elements, full_matrix_capture, simulation, voids
from tfm import tfm_kernel

BASE_DIR = Path(__file__).parent
RGB_PNG_DIR = (BASE_DIR / "../../results/rgb_png").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--pixels", type=int, default=512, help="image pixels per side")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
SPACE_ORDER = 8  # the measurement
DYNAMIC_RANGE = 30  # dB shown below the strongest reflection

# ------------------------------------ measurement ------------------------------------
sim, t, sources, sensors = simulation(SPACE_ORDER)
truth = voids(sim)
tic = time.time()
scattered = full_matrix_capture(sim, sources, sensors, truth)
scattered -= full_matrix_capture(sim, sources, sensors, cp.ones_like(truth))  # without voids
cp.cuda.Device().synchronize()
print(f"full matrix capture, {2 * len(elements)} shots of {sim.N} steps: {time.time() - tic:.2f} s")

# ---------------------------------------- image --------------------------------------
h = cp.ascontiguousarray(hilbert(scattered, axis=1).transpose(0, 2, 1)).astype(cp.complex64)
x, y = cp.meshgrid(cp.linspace(0, Lx, args.pixels), cp.linspace(0, Ly, args.pixels), indexing="ij")
t0 = CYCLES / (2 * FREQUENCY)  # center of the burst
image = tfm_kernel(h, elements, t0, t[1], WAVESPEED, x, y)  # compile
cp.cuda.Device().synchronize()
tic = time.time()
image = tfm_kernel(h, elements, t0, t[1], WAVESPEED, x, y)
cp.cuda.Device().synchronize()
print(f"{args.pixels}^2 pixels, {len(elements)}^2 traces: {(time.time() - tic) * 1e3:.2f} ms")

# ----------------------------------- postprocessing ----------------------------------
decibel = 20 * np.log10(image.get() / float(image.max()))
fig, ax = plt.subplots(figsize=(args.pixels / 150, args.pixels / 150), dpi=150)
ax.imshow(decibel.T, origin="lower", cmap="binary", vmin=-DYNAMIC_RANGE, vmax=0)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
else:
    plt.savefig(RGB_PNG_DIR / "plate_tfm.png")
