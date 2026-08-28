import argparse
import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import AcousticWave, ScalarWave, simulate, stable_dt

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = (BASE_DIR / "../../results/animations/animation_frames").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2
FORMULATION = "scalar"  # "scalar" or "acoustic"
PRECISION = "float32"  # "float32" or "float64"
SPACE_ORDER = 2  # finite difference order, any even number
THREADS = (4, 128) if DIM == 2 else (4, 4, 64) if DIM == 3 else (128,)

# physics
LENGTH = 1
WAVESPEED = 0.5
DENSITY = 1
AMPLITUDE = 1e8
FREQUENCY = 2  # bounded by WAVESPEED / (20.0 * min(dx))
CYCLES = 5
T = 20

RESOLUTION = 500 if DIM == 2 else 100 if DIM == 3 else 400
SAFETY = 0.95  # fraction of the stable time step
SAVE_EVERY = 10

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION,) * DIM
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
N = math.ceil(T / dt)

if FORMULATION == "acoustic":
    sim = AcousticWave(
        Nx, dx, N, dt, THREADS, precision=PRECISION, space_order=SPACE_ORDER,
        rho1=1.204, rho2=2643.0, kappa1=1.419e5, kappa2=6.87e8,
    )
else:
    sim = ScalarWave(
        Nx, dx, N, dt, THREADS, precision=PRECISION, space_order=SPACE_ORDER,
        wavespeed=WAVESPEED, density=DENSITY,
    )

indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)

# source in the origin corner, spread over the cell volume by point_source
t = np.linspace(0, (N - 1) * dt, N)
source = point_source(sim, (0.0,) * DIM, sineburst(t, AMPLITUDE, FREQUENCY, CYCLES))

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
if args.book and DIM == 1:
    record_every = 4
elif args.animate and DIM in (1, 2):
    record_every = SAVE_EVERY
else:
    record_every = None
result = simulate(sim, source, indicator, record_every=record_every)
u, snaps = result if record_every is not None else (result, None)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = u.get()
scale = float(np.max(np.abs(u_np)))

if not args.book and not args.animate:
    if DIM == 1:
        x = np.linspace(0, LENGTH, Nx[0])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, u_np, "k")
    else:
        field = u_np if DIM == 2 else u_np[Nx[0] // 2]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pcolormesh(field.T, cmap=cmr.fusion, vmin=-scale, vmax=scale)
        ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
if args.book:
    if DIM == 1:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.pcolormesh(snaps, cmap=cmr.fusion, vmin=-0.75 * scale, vmax=0.75 * scale)
        ax.axis("off")
        ax.set_rasterized(True)  # vectorized pdf too large at this grid resolution
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / "wave1D.pdf")
        plt.close()
# ----------------------------------- animate export ----------------------------------
if args.animate and DIM in (1, 2):
    frame_dir = ANIMATION_DIR / f"wave{DIM}D"
    frame_dir.mkdir(parents=True, exist_ok=True)
    anim_scale = float(np.max(np.abs(snaps)))
    if DIM == 1:
        x = np.linspace(0, LENGTH, Nx[0])
        for i, snap in enumerate(snaps):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(x, snap, "k")
            ax.set_ylim(-anim_scale, anim_scale)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(frame_dir / f"frame_{i}.jpg")
            plt.close()
    else:
        for i, snap in enumerate(snaps):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(
                snap.T,
                cmap=cmr.fusion,
                vmin=-0.6 * anim_scale,
                vmax=0.6 * anim_scale,
                origin="lower",
            )
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(frame_dir / f"frame_{i}.jpg")
            plt.close()
