import argparse
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import lbm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()
ANIMATION_DIR = (RESULTS_DIR / "animations/animation_frames/lbm_turbulence").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float16
TILE_X, TILE_Y, ROWS = 48, 64, 8  # as in temporal_step
STEPS_PER_LAUNCH = 6  # as STEPS in temporal_step
COMPILER_OPTIONS = ("--use_fast_math", "-Dstore=__half")

# physics
VELOCITY = 0.05  # rms velocity, Mach number 0.09
WAVELENGTH = 200  # nodes, of the most energetic vortices at the start
VISCOSITY = 0.008

# discretization
NX, NY = 8192, 4096
STEPS = 30000

# postprocessing
FRAMES = 250  # animation frames over all steps
DOWNSAMPLE = 4  # nodes per frame pixel in each direction

# ------------------------------------ prepare data -----------------------------------
omega_plus, omega_minus = lbm.relaxation_rates(VISCOSITY)
ux, uy = lbm.random_vortices(NX, NY, WAVELENGTH, VELOCITY)
f = lbm.equilibrium(ux, uy, DTYPE)
f_new = cp.empty_like(f)
del ux, uy

# --------------------------------------- setup ---------------------------------------
module = cp.RawModule(code=(BASE_DIR / "lbm.cu").read_text(), options=COMPILER_OPTIONS)
temporal = module.get_function("temporal_step")
# bytes: a guard row and the eight moving populations of the tile
SHARED = 4 * (8 * TILE_X * TILE_Y + 2 * (TILE_Y + 1))
temporal.max_dynamic_shared_size_bytes = SHARED
tiles = (
    (NX + TILE_X - 2 * STEPS_PER_LAUNCH - 1) // (TILE_X - 2 * STEPS_PER_LAUNCH),
    (NY + TILE_Y - 2 * STEPS_PER_LAUNCH - 1) // (TILE_Y - 2 * STEPS_PER_LAUNCH),
)


def lbm_steps(f, f_new):
    temporal(
        tiles,
        (TILE_Y, ROWS),
        (f, f_new, omega_plus, omega_minus, np.int32(NX), np.int32(NY)),
        shared_mem=SHARED,
    )
    return f_new, f


def vorticity(f):
    w = lbm.vorticity(*lbm.velocity(f))
    return (
        w.reshape(NX // DOWNSAMPLE, DOWNSAMPLE, NY // DOWNSAMPLE, DOWNSAMPLE)
        .mean((1, 3))
        .get()
    )


# --------------------------------------- solve ---------------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

lbm_steps(f.copy(), cp.empty_like(f))  # warm-up: the first launch compiles the kernel
cp.cuda.Stream.null.synchronize()
elapsed = 0.0
launches_per_frame = STEPS // STEPS_PER_LAUNCH // FRAMES
for frame in range(FRAMES):
    tic = time.time()
    for t in range(launches_per_frame):
        f, f_new = lbm_steps(f, f_new)
    cp.cuda.Stream.null.synchronize()
    elapsed += time.time() - tic
    if args.animate:
        w = vorticity(f)
        scale = 0.6 * np.percentile(abs(w), 99.5)
        plt.imsave(
            ANIMATION_DIR / f"frame_{frame:04d}.jpg",
            w.T[::-1],
            cmap=cmr.guppy,
            vmin=-scale,
            vmax=scale,
        )

steps = FRAMES * launches_per_frame * STEPS_PER_LAUNCH
print(
    f"elapsed time {elapsed:.2f} s for {steps} steps ({elapsed / steps * 1e3:.4f} ms/step)"
)
print(f"{NX * NY / (elapsed / steps) / 1e9:.2f} billion node updates/s")

# ----------------------------------- postprocessing ----------------------------------
w = vorticity(f)
scale = 0.6 * np.percentile(abs(w), 99.5)  # the strongest vortex cores saturate

if not args.book:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.imshow(w.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.imsave(
        RGB_PNG_DIR / "lbm_turbulence.png",
        w.T[::-1],
        cmap=cmr.guppy,
        vmin=-scale,
        vmax=scale,
    )
