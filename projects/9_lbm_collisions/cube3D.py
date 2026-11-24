import math
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuvox.core.graphs import define_march
from cuvox.core.lattice import grid_coords
from cuvox.lbm.boundary import Outflow, Periodic, Wall
from cuvox.lbm.cumulant import Cumulant
from cuvox.lbm.simulation import (
    LatticeBoltzmann,
    define_simulation,
    equilibrium,
    moments,
)
from cuvox.lbm.velocities import D3Q27

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
# physics
REYNOLDS = 1e4  # based on the cube edge and the inflow velocity

# geometry, in cube edges
BOX = 6  # edge of the cubic domain
CENTER = 1.5  # distance of the cube center from the inlet

# discretization
SIZE = 64  # nodes across the cube
VELOCITY = 0.05  # inflow in lattice units, Mach number 0.09
FLOWTHROUGHS = 2  # domain lengths travelled at the inflow velocity
CHECKS = 400  # blow-up checks over the run of a single-rate or two-rate collision
THREADS = (1, 4, 64)

# --------------------------------------- setup ---------------------------------------
Nx = (BOX * SIZE + 2,) * 3
steps = math.ceil(FLOWTHROUGHS * BOX * SIZE / VELOCITY)
x, y, z = grid_coords(Nx)
center = (CENTER * SIZE, BOX * SIZE / 2, BOX * SIZE / 2)
solid = (abs(x - center[0]) <= SIZE / 2) & (abs(y - center[1]) <= SIZE / 2)
solid &= abs(z - center[2]) <= SIZE / 2
del x, y, z
inner = tuple(slice(1, n - 1) for n in Nx)
fluid = ~solid[inner]
options = dict(
    precision="float32",
    viscosity=VELOCITY * SIZE / REYNOLDS,
    lattice=D3Q27,
    boundary=((Wall(velocity=(VELOCITY, 0.0, 0.0)), Outflow()), Periodic(), Periodic()),
    solid=solid,
)


# --------------------------------------- helper --------------------------------------
def build(collision):
    if collision == "cumulant":
        return Cumulant(Nx, steps, THREADS, **options)
    magic = 9 * options["viscosity"] ** 2 if collision == "BGK" else 0.25  # BGK: equal rates
    return LatticeBoltzmann(Nx, steps, THREADS, magic=magic, **options)


def run(collision, checks):
    sim = build(collision)
    f, step = define_simulation(sim, equilibrium(sim, 1.0, (VELOCITY, 0.0, 0.0)))
    advance = define_march(step)
    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    t = 0
    for k in range(1, checks + 1):
        advance(t, k * steps // checks - t)
        t = k * steps // checks
        speed = cp.sqrt((moments(sim, f[t % 2]).u ** 2).sum(0))
        if bool(cp.isnan(speed[fluid]).any()) or float(cp.nanmax(speed)) > 10 * VELOCITY:
            return None, t, None
    cp.cuda.Stream.null.synchronize()
    toc = time.time()
    return moments(sim, f[t % 2]).u, t, toc - tic


# --------------------------------------- solve ---------------------------------------
for collision in ("BGK", "TRT"):
    field, blown, _ = run(collision, CHECKS)
    print(f"{collision}: {'stable' if field is not None else 'blew up'} at step {blown} of {steps}")
u, _, seconds = run("cumulant", 10)
print(f"cumulant: {steps} steps in {seconds:.2f} s, {27 * (BOX * SIZE) ** 3:.2e} dofs")

# ----------------------------------- postprocessing ----------------------------------
u = cp.nan_to_num(u)
grad = [[cp.gradient(u[i], axis=j) for j in range(3)] for i in range(3)]
q = -0.5 * sum(grad[i][j] * grad[j][i] for i in range(3) for j in range(3))
vorticity_x = grad[2][1] - grad[1][2]
np.savez_compressed(
    DATA_DIR / "lbm_cube.npz",
    q=(q * (SIZE / VELOCITY) ** 2).astype(cp.float16).get(),
    vorticity_x=(vorticity_x * SIZE / VELOCITY).astype(cp.float16).get(),
    solid=solid[inner].get(),
)
print(f"saved {DATA_DIR / 'lbm_cube.npz'}")
