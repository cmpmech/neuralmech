import argparse
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuvox.core.graphs import define_march
from cuvox.core.lattice import grid_coords
from cuvox.lbm.boundary import Periodic
from cuvox.lbm.cumulant import Cumulant
from cuvox.lbm.simulation import (
    LatticeBoltzmann,
    define_simulation,
    define_temporal_block,
    equilibrium,
    moments,
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# physics, thin shear layer of Minion and Brown (1997) in the periodic unit box
REYNOLDS = 1e5
WIDTH = 80  # inverse layer thickness
PERTURBATION = 0.05  # cross velocity over shear velocity

# discretization
VELOCITY = 0.1  # shear velocity in lattice units, Mach number 0.17
RUNS = (("BGK", 256), ("TRT", 256), ("cumulant", 256), ("cumulant", 1024))
TIMES = (0.55, 1.0)  # box lengths travelled at the shear velocity
CHECKS = 40  # blow-up checks over the run
RESOLUTION = 256  # pixels per side of the saved fields
THREADS = (4, 128)


# --------------------------------------- helper --------------------------------------
def build(collision, L, steps):
    options = dict(
        precision="float32", viscosity=VELOCITY * L / REYNOLDS, boundary=Periodic()
    )
    if collision == "cumulant":
        return Cumulant((L + 2,) * 2, steps, THREADS, **options)
    magic = 9 * options["viscosity"] ** 2 if collision == "BGK" else 0.25  # BGK: equal rates
    return LatticeBoltzmann((L + 2,) * 2, steps, THREADS, magic=magic, **options)


def vorticity(u, L):
    du_y = (cp.roll(u[1], -1, 0) - cp.roll(u[1], 1, 0)) / 2
    du_x = (cp.roll(u[0], -1, 1) - cp.roll(u[0], 1, 1)) / 2
    return (du_y - du_x) * L / VELOCITY


def run(collision, L):
    steps = round(L / VELOCITY)
    sim = build(collision, L, steps)
    x, y = ((a + 0.5) / L for a in grid_coords((L + 2,) * 2))
    layers = cp.where(y <= 0.5, cp.tanh(WIDTH * (y - 0.25)), cp.tanh(WIDTH * (0.75 - y)))
    u0 = VELOCITY * cp.stack((layers, PERTURBATION * cp.sin(2 * np.pi * (x + 0.25))))
    f, step = define_simulation(sim, equilibrium(sim, 1.0, u0))
    advance = define_march(step, define_temporal_block(sim, f))

    fields, t = {}, 0
    for k in range(1, CHECKS + 1):
        advance(t, k * steps // CHECKS - t)
        t = k * steps // CHECKS
        u = moments(sim, f[t % 2]).u
        if not float(cp.abs(u).max()) < 10 * VELOCITY:
            return fields, k / CHECKS
        if k / CHECKS in TIMES:
            pooled = vorticity(u, L).reshape(RESOLUTION, L // RESOLUTION, RESOLUTION, -1)
            fields[k / CHECKS] = pooled.mean((1, 3)).get()
    return fields, None


# --------------------------------------- solve ---------------------------------------
results = {}
for collision, L in RUNS:
    fields, blown = run(collision, L)
    results[collision, L] = fields
    state = "stable" if blown is None else f"blew up by t = {blown:.3f}"
    print(f"{collision} on {L}^2 nodes: {state}")

# ----------------------------------- postprocessing ----------------------------------
reference = results["cumulant", 1024]
scales = {t: 0.5 * np.percentile(abs(reference[t]), 99.5) for t in TIMES}

if not args.book:
    fig, axes = plt.subplots(len(TIMES), len(RUNS), figsize=(12, 6))
    for i, t in enumerate(TIMES):
        for j, run_key in enumerate(RUNS):
            axes[i, j].axis("off")
            if t in results[run_key]:
                axes[i, j].imshow(
                    results[run_key][t].T,
                    origin="lower",
                    cmap=cmr.guppy,
                    vmin=-scales[t],
                    vmax=scales[t],
                )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for (collision, L), fields in results.items():
        for t, w in fields.items():
            plt.imsave(
                RGB_PNG_DIR / f"lbm_shear_{collision}_{L}_t{round(100 * t):03d}.png",
                w.T[::-1],
                cmap=cmr.guppy,
                vmin=-scales[t],
                vmax=scales[t],
            )
