import argparse
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
from cuwave.evals import f1_score, l2_error
from cuwave.optimization import Adam
from cuwave.utils import interior_slice, misfit_gradient, resample

from plate import GAMMA_VOID, full_matrix_capture, simulation, voids

BASE_DIR = Path(__file__).parent
RGB_PNG_DIR = (BASE_DIR / "../../results/rgb_png").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
SPACE_ORDER, SPACE_ORDER_OBS = 4, 8  # inversion and measurement
SHOTS = slice(None, None, 4)  # every fourth element fires, all listen
ITERATIONS, LR = 60, 0.1
GAMMA_MIN = 1e-3

# ------------------------------------ measurement ------------------------------------
sim_obs, t_obs, sources_obs, sensors_obs = simulation(SPACE_ORDER_OBS)
sim, t, sources, sensors = simulation(SPACE_ORDER)
truth = voids(sim_obs)
observed = full_matrix_capture(sim_obs, sources_obs[SHOTS], sensors_obs, truth)
observed = [resample(record, t_obs[1], t[1], sim.N) for record in observed]

# ------------------------------------ optimization -----------------------------------
gamma = cp.ones(sim.Nx_padded, dtype=sim.dtype)
optimizer = Adam(lr=LR)
misfit_gradient(sim, sources[SHOTS][:1], gamma, sensors, observed[:1])  # compile
cp.cuda.Device().synchronize()
tic = time.time()
for iteration in range(ITERATIONS):
    cost, gradient = misfit_gradient(sim, sources[SHOTS], gamma, sensors, observed)
    gamma = cp.clip(optimizer.step(gamma, gradient), GAMMA_MIN, 1.0)
cp.cuda.Device().synchronize()
elapsed = time.time() - tic
print(f"{len(observed)} shots of {sim.N} steps, {ITERATIONS} iterations: {elapsed:.2f} s, "
      f"{elapsed / ITERATIONS * 1e3:.0f} ms per iteration")

inner = interior_slice(sim)
print(f"relative L2 error {l2_error(gamma[inner], truth[inner]):.3f}, "
      f"f1 at gamma < 0.5 {f1_score(gamma[inner], truth[inner], threshold=0.5):.3f}")

# ----------------------------------- postprocessing ----------------------------------
for name, field in (("plate_truth", truth), ("plate_fwi", gamma)):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
    ax.imshow(1 - field[inner].get().T, origin="lower", cmap="binary", vmin=0, vmax=1 - GAMMA_VOID)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if args.book:
        plt.savefig(RGB_PNG_DIR / f"{name}.png")
if not args.book:
    plt.show()
