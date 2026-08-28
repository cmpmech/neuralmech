import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import numpy as np
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import ScalarWave, simulate, stable_dt
from PIL import Image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
ANIMATION_DIR = (BASE_DIR / "../../results/animations/animation_frames").resolve()

# -------------------------------------- settings -------------------------------------
SPACE_ORDER = 2  # finite difference order, any even number
SAFETY = 0.95  # fraction of the stable time step
THREADS = (4, 128)

# physics
WAVESPEED = 6000.0
DENSITY = 2700.0
AMPLITUDE = 1e8
POINTS_PER_WAVELENGTH = 10
CYCLES = 1
T = 1e-5
MIN_INDICATOR = 1e-3

# animation
SAVE_EVERY = 2
SCALE = 5e-6  # fixed color range, so the frames share one scale
VOID_GREY = 178  # Greys_r at 0.7

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
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
frequency = WAVESPEED / (POINTS_PER_WAVELENGTH * dx[0])
N = math.ceil(T / dt)

sim = ScalarWave(
    (Nx, Ny), dx, N, dt, THREADS, space_order=SPACE_ORDER,
    wavespeed=WAVESPEED, density=DENSITY,
)

# source in the origin corner, spread over the cell volume by point_source
t = np.linspace(0, (N - 1) * dt, N)
source = point_source(sim, (0.0, 0.0), sineburst(t, AMPLITUDE, frequency, CYCLES))

indicator_padded = cp.ones(sim.Nx_padded, dtype=sim.dtype)
indicator_padded[:Nx, :Ny] = cp.asarray(indicator, dtype=sim.dtype)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
_, snaps = simulate(sim, source, indicator_padded, record_every=SAVE_EVERY)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- animate export ----------------------------------
frame_dir = ANIMATION_DIR / "bhai_waves"
frame_dir.mkdir(parents=True, exist_ok=True)
void = indicator == MIN_INDICATOR

for i, snap in enumerate(snaps):
    # normalize wave field to [0, 1] and apply colormap
    normalized = np.clip((snap + SCALE) / (2 * SCALE), 0, 1)
    rgb = (cmr.fusion(normalized)[:, :, :3] * 255).astype(np.uint8)
    rgb[void] = VOID_GREY
    Image.fromarray(rgb, mode="RGB").save(frame_dir / f"frame_{i}.jpg", quality=85)
print(f"saved {len(snaps)} frames to {frame_dir}")
