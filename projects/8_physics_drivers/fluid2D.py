import argparse
import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cufluid.geometry import circle
from cufluid.grid import grid_coords
from cufluid.lbm.boundary import Outflow, Wall
from cufluid.lbm.lbm import LatticeBoltzmann, moments, simulate

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")  # TODO could be animated
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# discretization
DIAMETER = 60  # lattice nodes across the cylinder
VELOCITY = 0.05  # inflow in lattice units, Mach number 0.09
THREADS = (4, 128)

# physics
REYNOLDS = 100
T = 80  # flow-through times of one diameter

# geometry, in diameters
LENGTH, HEIGHT = 24, 4
CENTER = (2, 1.95)  # off the channel axis, so the shedding starts by itself

# --------------------------------------- setup ---------------------------------------
Nx = (round(LENGTH * DIAMETER) + 2, round(HEIGHT * DIAMETER) + 2)
viscosity = VELOCITY * DIAMETER / REYNOLDS
N = math.ceil(T * DIAMETER / VELOCITY)

center = [c * DIAMETER for c in CENTER]
solid = circle(grid_coords(Nx), center, DIAMETER / 2)

# inflow through a moving wall on the left, no-slip walls on top and bottom
inlet = Wall(velocity=(VELOCITY, 0.0))
sim = LatticeBoltzmann(
    Nx,
    N,
    THREADS,
    viscosity=viscosity,
    boundary=((inlet, Outflow()), Wall()),
    solid=solid,
)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
f = simulate(sim, u0=(VELOCITY, 0.0))
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
rho, u = moments(sim, f)
u = u.get()
vorticity = np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)
scale = 0.2 * np.nanmax(np.abs(vorticity))

fig, ax = plt.subplots(figsize=(LENGTH, HEIGHT), dpi=100)
ax.pcolormesh(vorticity.T, cmap=cmr.guppy, vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / "cylinderflow.pdf", transparent=True)
    plt.close()
