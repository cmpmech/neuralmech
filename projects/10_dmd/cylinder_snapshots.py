import math
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cufluid.geometry import circle
from cufluid.grid import grid_coords
from cufluid.lbm.boundary import Outflow, Wall
from cufluid.lbm.lbm import LatticeBoltzmann, simulate

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
# discretization
DIAMETER = 24  # lattice nodes across the cylinder
VELOCITY = 0.05  # inflow in lattice units, Mach number 0.09
THREADS = (4, 128)
RECORD_EVERY = 100  # steps between snapshots

# physics
REYNOLDS = 100
T = 120  # flow-through times of one diameter

# geometry, in diameters
LENGTH, HEIGHT = 16, 4
CENTER = (2, 1.95)  # off the channel axis, so the shedding starts by itself
CROP = (1, 11)  # stored window along the channel

# --------------------------------------- setup ---------------------------------------
Nx = (round(LENGTH * DIAMETER) + 2, round(HEIGHT * DIAMETER) + 2)
viscosity = VELOCITY * DIAMETER / REYNOLDS
N = math.ceil(T * DIAMETER / VELOCITY)

center = [c * DIAMETER for c in CENTER]
solid = circle(grid_coords(Nx), center, DIAMETER / 2)

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
_, u = simulate(sim, u0=(VELOCITY, 0.0), record_every=RECORD_EVERY)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
nodes = u.shape[2] * u.shape[3]
print(f"{nodes} nodes, {9 * nodes} populations, {N} steps")

# --------------------------------------- export --------------------------------------
crop = slice(CROP[0] * DIAMETER, CROP[1] * DIAMETER)
vorticity = np.gradient(u[:, 1], axis=1) - np.gradient(u[:, 0], axis=2)
vorticity = np.nan_to_num(vorticity[:, crop] * DIAMETER / VELOCITY).astype(np.float32)
mask = np.isnan(u[0, 0, crop])
dt = RECORD_EVERY * VELOCITY / DIAMETER  # snapshot spacing in convective time D/U
t = np.arange(len(vorticity)) * dt

np.savez(DATA_DIR / "cylinder_snapshots.npz", vorticity=vorticity, mask=mask, t=t)
print(f"saved {vorticity.shape} to {DATA_DIR / 'cylinder_snapshots.npz'}")
