import argparse
import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from solvers.scalar_wave import setup_simulation, setup_source, simulate2D

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------- settings ---------------------------
LX = 0.05
LY = 0.05
WAVESPEED = 6000.
DENSITY = 2700.
NX = 500
NY = 500
AMPLITUDE = 1e8
CYCLES = 3
T = 1e-5
THREADS = (4, 128)

# ----------------------- simulation setup ----------------------
dx = LX / (NX - 3)
dy = LY / (NY - 3)
dt = 0.95 * min(dx, dy) / WAVESPEED / math.sqrt(2)
frequency = WAVESPEED / (20. * dx)
N = math.ceil(T / dt)

sim = setup_simulation((NX, NY), (dx, dy), N, dt, WAVESPEED, DENSITY, THREADS)

indicator = cp.ones(sim.Nx_padded, dtype=cp.float32)

# ---------------------------- source ---------------------------
def sineburst(t, amplitude, frequency, cycles):
    mask = (t > 0) & (t <= cycles / frequency)
    return amplitude * mask * np.sin(2 * np.pi * frequency * t) \
           * np.sin(np.pi * frequency * t / cycles) ** 2

t_arr = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_arr, AMPLITUDE, frequency, CYCLES) / (dx * dy)
signal = cp.asarray(signal_np[:, None], dtype=cp.float32)

source_pos = cp.array([[NX // 2], [NY // 2]], dtype=cp.int32)
source = setup_source(source_pos, signal)

# ----------------------------- solve ---------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate2D(sim, source, indicator, precompiled=False)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed: {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ---------------------------- plot -----------------------------
x_1d = np.linspace(0, LX, NX - 2)
y_1d = np.linspace(0, LY, NY - 2)
u_np = u.get()[1:-1, 1:-1]
scale = float(np.max(np.abs(u_np)))

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(x_1d, y_1d, u_np.T, cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.set_aspect("equal")

plt.tight_layout()

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "scalar_wave2D_example.pdf")
    plt.close()
elif args.animate:
    plt.close()
else:
    plt.show()
