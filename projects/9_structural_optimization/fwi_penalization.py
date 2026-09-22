import argparse
import math
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuwave.evals import f1_score, l2_error
from cuwave.geometry import circle
from cuwave.optimization import Lbfgs
from cuwave.regularization import TotalVariation
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import (
    Sensors,
    interior_slice,
    line,
    measure,
    misfit,
    misfit_gradient,
    resample,
    shots,
)
from cuwave.wave import grid_coords, stable_dt

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# discretization: the measurement gets its own grid and stencil, against the inverse crime
RESOLUTION, SPACE_ORDER = (256, 256), 4
RESOLUTION_OBS, SPACE_ORDER_OBS = (383, 383), 8
PRECISION = "float32"
SAFETY = 0.99  # fraction of the stable time step
GAMMA_MIN = 1e-3

# physics
T = 4.0  # traversals of the plate height
WAVESPEED, DENSITY = 1.0, 1.0
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 20.0, 2
# transducer along the top edge
NUM_SOURCES, NUM_SENSORS = 4, 32
ARRAY_SPAN = (0.1, 0.9)

# geometry: unit plate with a central circular hole
LENGTHS = (1.0, 1.0)
RADIUS = 0.15
GAMMA_VOID = 1e-5

# penalization
WEIGHT = 3e-3
TV_EPS = 1e-2  # below this gradient magnitude total variation acts like Tikhonov

# optimization
ITERS, PAIRS = 40, 4  # secant pairs L-BFGS keeps
FIRST_STEP = 0.05
ARMIJO = 1e-4  # sufficient decrease the line search asks for
BACKTRACK = 0.5  # factor a rejected trial step shrinks by
MIN_ALPHA = 1e-3  # give up below this fraction of the proposed step

# evaluation
THRESHOLD = 0.5

# --------------------------------------- setup ---------------------------------------
Lx, Ly = LENGTHS


def simulation(resolution, space_order):
    dx = tuple(LENGTHS[d] / (resolution[d] - 3) for d in range(len(resolution)))
    N = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, space_order))) + 1
    return ScalarWave(
        resolution,
        dx,
        N,
        T / (N - 1),
        (4, 64),
        precision=PRECISION,
        space_order=space_order,
        wavespeed=WAVESPEED,
        density=DENSITY,
    )


def plate_with_hole(sim):
    coords = grid_coords(sim.Nx, sim.dx, dtype=sim.dtype)
    hole = circle(coords, (0.5 * Lx, 0.5 * Ly), RADIUS)
    return cp.where(hole, GAMMA_VOID, 1.0).astype(sim.dtype)


sim = simulation(RESOLUTION, SPACE_ORDER)
sim_obs = simulation(RESOLUTION_OBS, SPACE_ORDER_OBS)

surface = lambda count: line((ARRAY_SPAN[0], Ly), (ARRAY_SPAN[1], Ly), count)
source_coords, sensor_coords = surface(NUM_SOURCES), surface(NUM_SENSORS)
burst = lambda t: sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)

print(
    f"{WAVESPEED / (FREQUENCY * max(sim.dx)):.0f} points per wavelength, "
    f"{RADIUS / max(sim.dx):.1f} per hole radius"
)

# ------------------------------------ measurement ------------------------------------
sources_obs = shots(sim_obs, source_coords, burst(np.linspace(0, T, sim_obs.N)))
sensors_obs = Sensors(sim_obs, sensor_coords)
truth_obs = plate_with_hole(sim_obs)

tic = time.time()
observed = [
    resample(record, sim_obs.dt, sim.dt, sim.N)
    for record in measure(sim_obs, sources_obs, truth_obs, sensors_obs)
]
cp.cuda.Stream.null.synchronize()
print(f"elapsed time {time.time() - tic:.2f} s")

# ------------------------------------ optimization -----------------------------------
sources = shots(sim, source_coords, burst(np.linspace(0, T, sim.N)))
sensors = Sensors(sim, sensor_coords)
truth = plate_with_hole(sim)


def invert(penalty):
    gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    optimizer = Lbfgs(
        k=PAIRS,
        first_step=FIRST_STEP,
        armijo=ARMIJO,
        shrink=BACKTRACK,
        min_alpha=MIN_ALPHA,
    )
    roughness = (lambda g: 0.0) if penalty is None else (lambda g: float(penalty(g)))
    # the line search re-evaluates the full objective, so the penalty enters it too
    objective = lambda g: (
        float(misfit(sim, sources, g, sensors, observed)) + roughness(g)
    )
    misfits = []
    for iteration in range(ITERS):
        cost, gradient = misfit_gradient(sim, sources, gamma, sensors, observed)
        misfits.append(float(cost))
        if penalty is not None:
            gradient += penalty.grad(gamma)
        gamma = optimizer.search(
            gamma,
            gradient,
            cost + roughness(gamma),
            objective,
            lambda g: cp.clip(g, GAMMA_MIN, 1.0),
        )[0]
    misfits.append(float(misfit(sim, sources, gamma, sensors, observed)))
    return gamma, misfits


penalties = [None, TotalVariation(WEIGHT, eps=TV_EPS)]
names = ["fwi_plain", "fwi_total_variation"]

tic = time.time()
runs = [invert(penalty) for penalty in penalties]
reconstructions = [gamma for gamma, _ in runs]
cp.cuda.Stream.null.synchronize()
print(f"elapsed time {time.time() - tic:.2f} s")

# ------------------------------------- evaluation ------------------------------------
interior = interior_slice(sim)
reference = truth[interior]

for name, (gamma, misfits) in zip(names, runs):
    print(
        f"{name}: misfit {misfits[-1]:.3e} ({misfits[-1] / misfits[0]:.3f} of the "
        f"start)  f1 {f1_score(gamma[interior], reference, threshold=THRESHOLD):.3f}  "
        f"relative L2 error {l2_error(gamma[interior], reference):.3f}"
    )

# ----------------------------------- postprocessing ----------------------------------
fields = [truth] + reconstructions
filenames = ["fwi_truth"] + names
show = lambda ax, field: ax.imshow(
    field[interior].get().T, origin="lower", cmap="binary_r", vmin=0.0, vmax=1.0
)

if not args.book:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, field, filename in zip(axes, fields, filenames):
        show(ax, field)
        ax.set_title(filename)
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for filename, field in zip(filenames, fields):
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        show(ax, field)
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"{filename}.pdf")
        plt.close(fig)
