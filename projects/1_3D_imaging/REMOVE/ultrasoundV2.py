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
LX = 0.1
LY = 0.1
WAVESPEED = 3000.0
DENSITY = 1000.0
NX = 300
NY = 300
AMPLITUDE = 1e6
CYCLES = 3
T = 0.6e-4  # 1.2e-4
THREADS = (4, 128)
SENSOR_X_IDX = [2, 3, 4, 5, 6, 7]
EMITTER_IDX = 0
REFLECTOR_INDICATOR = 5.0
REFLECTOR_HALF = 10

# ----------------------- simulation setup ----------------------
dx = LX / (NX - 3)
dy = LY / (NY - 3)
dt = 0.95 * min(dx, dy) / WAVESPEED / math.sqrt(2)
frequency = WAVESPEED / (20.0 * dx)
N = math.ceil(T / dt)

sim = setup_simulation((NX, NY), (dx, dy), N, dt, WAVESPEED, DENSITY, THREADS)

# ------------------- domain + reflectivity --------------------
# rows = x-direction, cols = y-direction (matches scalar_wave convention)
# physical x: from 0 (row 1) to LX (row NX-2)
# physical y: from 0 (col 1) to LY (col NY-2)
# sensors are at y=0 → col 1, at various x positions → various rows

x10 = np.linspace(0, 1, 10)
sensor_xs = x10[SENSOR_X_IDX] * LX
emitter_x = sensor_xs[EMITTER_IDX]

sensor_rows = np.round(sensor_xs / dx).astype(int) + 1
sensor_cols = np.ones(len(SENSOR_X_IDX), dtype=int)
emitter_row = int(np.round(emitter_x / dx)) + 1

# reflector block at relative position (4/9, 6/9) matching ultrasound.py z[4,6]
i_r = round(4 / 9 * (NX - 3)) + 1
j_r = round(6 / 9 * (NY - 3)) + 1

indicator_np = np.ones((NX, NY), dtype=np.float32)
indicator_np[
    i_r - REFLECTOR_HALF : i_r + REFLECTOR_HALF,
    j_r - REFLECTOR_HALF : j_r + REFLECTOR_HALF,
] = REFLECTOR_INDICATOR

indicator = cp.ones(sim.Nx_padded, dtype=cp.float32)
indicator[:NX, :NY] = cp.asarray(indicator_np)


# ---------------------------- source ---------------------------
def sineburst(t, amplitude, frequency, cycles):
    mask = (t > 0) & (t <= cycles / frequency)
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * frequency * t)
        * np.sin(np.pi * frequency * t / cycles) ** 2
    )


t_arr = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_arr, AMPLITUDE, frequency, CYCLES) / (dx * dy)
signal = cp.asarray(signal_np[:, None], dtype=cp.float32)

source_pos = cp.array([[emitter_row], [1]], dtype=cp.int32)
source = setup_source(source_pos, signal)

sensors = (
    cp.array(sensor_rows, dtype=cp.int32),
    cp.array(sensor_cols, dtype=cp.int32),
)

# ----------------------------- solve ---------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u_final, um = simulate2D(sim, source, indicator, sensors=sensors, precompiled=False)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed: {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

signals = um.get().T  # shape (n_sensors, N)

# mute direct wave with cosine taper centered on direct arrival
burst_duration = CYCLES / frequency
for k in range(len(SENSOR_X_IDX)):
    tof_direct = abs(sensor_xs[k] - emitter_x) / WAVESPEED
    dt_rel = t_arr - tof_direct
    in_window = np.abs(dt_rel) < burst_duration
    taper = np.where(in_window, 0.5 * (1 - np.cos(np.pi * dt_rel / burst_duration)), 1.0)
    signals[k] *= taper

# ---------------------------- delay-and-sum -----------------------------
x_1d = np.linspace(0, LX, NX - 2)
y_1d = np.linspace(0, LY, NY - 2)
x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing="ij")  # (NX-2, NY-2)

das = np.zeros((NX - 2, NY - 2))
for k, sx in enumerate(sensor_xs):
    d_emit = np.sqrt((x_grid - emitter_x) ** 2 + y_grid**2)
    d_recv = np.sqrt((x_grid - sx) ** 2 + y_grid**2)
    tofs = (d_emit + d_recv) / WAVESPEED
    das += np.interp(tofs.ravel(), t_arr, signals[k]).reshape(NX - 2, NY - 2)

# ---------------------------- plot -----------------------------
ind_plot = indicator_np[1:-1, 1:-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.pcolormesh(x_1d, y_1d, ind_plot.T, cmap="binary")
for k, (sx, sr) in enumerate(zip(sensor_xs, sensor_rows)):
    fc = "tab:red" if k == EMITTER_IDX else "tab:blue"
    ax.scatter(sx, 0.0, color=fc, zorder=5, s=40, clip_on=False)
ax.set_aspect("equal")

ax = axes[1]
vmax = float(np.max(np.abs(signals))) or 1.0
ax.imshow(
    signals,
    aspect="auto",
    extent=[0, T, len(SENSOR_X_IDX) - 0.5, -0.5],
    cmap="seismic",
    vmin=-vmax,
    vmax=vmax,
)

ax = axes[2]
vmax_das = float(np.max(np.abs(das))) or 1.0
ax.pcolormesh(x_1d, y_1d, das.T, cmap="seismic", vmin=-vmax_das, vmax=vmax_das)
ax.set_aspect("equal")

plt.tight_layout()

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "ultrasoundV2.pdf")
    plt.close()
elif args.animate:
    plt.close()
else:
    plt.show()
