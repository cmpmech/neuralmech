import argparse
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import lbm  # run with PYTHONPATH=.. from this folder

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../../results").resolve()
ANIMATION_DIR = (RESULTS_DIR / "animations/animation_frames/lbm_packing").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float16
TILE_X, TILE_Y, ROWS = 48, 64, 8  # as in temporal_step
STEPS_PER_LAUNCH = 6  # as STEPS in temporal_step
THREADS = 128
COMPILER_OPTIONS = ("--use_fast_math", "-Dstore=__half")

# physics
VELOCITY = 0.04  # inflow, Mach number 0.07
VISCOSITY = 0.012
SPONGE = 0.25  # largest relaxation toward the inflow per launch

# geometry
NX, NY = 16384, 4096
PACKING = (0.08, 0.3)  # streamwise extent of the packing, fraction of NX
RADII = (24, 64)
SOLID_FRACTION = 0.25
GAP = 12  # nodes between neighboring cylinders at least
OUTFLOW = 0.92  # the sponge starts here, fraction of NX

# discretization
STEPS = 307200

# postprocessing
FRAMES = 640
DOWNSAMPLE = 4  # nodes per frame pixel in each direction


# --------------------------------------- helper --------------------------------------
def random_packing(rng):
    """non-overlapping cylinders, periodic in y, until the solid fraction of the packing is reached."""
    solid = np.zeros((NX, NY), bool)
    a, b = (int(p * NX) for p in PACKING)
    placed, area = [], 0.0
    while area < SOLID_FRACTION * (b - a) * NY:
        r = rng.uniform(*RADII)
        cx, cy = rng.uniform(a + r, b - r), rng.uniform(0, NY)
        if all(
            (cx - x) ** 2 + (cy - y) ** 2 > (r + s + GAP) ** 2
            for x, y, s in placed[-400:]
        ):
            placed.append((cx, cy, r))
            area += np.pi * r**2
            xs = np.arange(int(cx - r), int(cx + r) + 2)
            ys = np.arange(int(cy - r), int(cy + r) + 2)
            solid[np.ix_(xs, ys % NY)] |= (xs[:, None] - cx) ** 2 + (
                ys[None, :] - cy
            ) ** 2 < r**2
    return cp.asarray(solid)


def wall_mask(solid):
    """bit i set where the node x - c_i is solid, so bit 0 marks a solid node itself."""
    mask = cp.zeros(solid.shape, cp.uint16)
    for i, c in enumerate(lbm.C):
        mask |= cp.roll(solid, tuple(c), axis=(0, 1)).astype(cp.uint16) << i
    return mask


# ------------------------------------ prepare data -----------------------------------
omega_plus, omega_minus = lbm.relaxation_rates(VISCOSITY)
solid = random_packing(np.random.default_rng(0))
wall = wall_mask(solid)
f = lbm.equilibrium(
    cp.full((NX, NY), VELOCITY, cp.float32), cp.zeros((NX, NY), cp.float32), DTYPE
)
f_new = cp.empty_like(f)

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR.parent / "lbm.cu").read_text() + (BASE_DIR / "walls.cu").read_text()
module = cp.RawModule(code=source, options=COMPILER_OPTIONS)
temporal = module.get_function("temporal_walls_step")
sponge = module.get_function("sponge")
SHARED = 4 * (8 * TILE_X * TILE_Y + 2 * (TILE_Y + 1))
temporal.max_dynamic_shared_size_bytes = SHARED
tiles = (
    (NX + TILE_X - 2 * STEPS_PER_LAUNCH - 1) // (TILE_X - 2 * STEPS_PER_LAUNCH),
    (NY + TILE_Y - 2 * STEPS_PER_LAUNCH - 1) // (TILE_Y - 2 * STEPS_PER_LAUNCH),
)
start = int(OUTFLOW * NX)
sponge_blocks = ((NY + THREADS - 1) // THREADS, NX - start)


def lbm_steps(f, f_new):
    temporal(
        tiles,
        (TILE_Y, ROWS),
        (f, f_new, wall, omega_plus, omega_minus, np.int32(NX), np.int32(NY)),
        shared_mem=SHARED,
    )
    sponge(
        sponge_blocks,
        (THREADS,),
        (
            f_new,
            np.float32(VELOCITY),
            np.float32(SPONGE),
            np.int32(start),
            np.int32(NX),
            np.int32(NY),
        ),
    )
    return f_new, f


def vorticity(f):
    ux, uy = lbm.velocity(f)
    w = cp.where(solid, 0, lbm.vorticity(ux, uy))
    return (
        w.reshape(NX // DOWNSAMPLE, DOWNSAMPLE, NY // DOWNSAMPLE, DOWNSAMPLE)
        .mean((1, 3))
        .get()
    )


def image(w):
    scale = 0.6 * np.percentile(abs(w), 99.5)
    rgb = cmr.guppy(np.clip(w / scale, -1, 1) * 0.5 + 0.5)
    rgb[solid[::DOWNSAMPLE, ::DOWNSAMPLE].get()] = (0.25, 0.25, 0.25, 1)
    return rgb.transpose(1, 0, 2)[::-1]


# --------------------------------------- solve ---------------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

cp.cuda.Stream.null.synchronize()
elapsed, t = 0.0, 0
launches_per_frame = STEPS // STEPS_PER_LAUNCH // FRAMES
for frame in range(FRAMES):
    tic = time.time()
    for _ in range(launches_per_frame):
        f, f_new = lbm_steps(f, f_new)
        t += STEPS_PER_LAUNCH
    cp.cuda.Stream.null.synchronize()
    elapsed += time.time() - tic
    if args.animate:
        plt.imsave(ANIMATION_DIR / f"frame_{frame:04d}.jpg", image(vorticity(f)))

print(
    f"elapsed time {elapsed:.2f} s for {t} steps ({NX * NY * t / elapsed / 1e9:.2f} billion node updates/s)"
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(16, 4), dpi=100)
ax.imshow(image(vorticity(f)))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
