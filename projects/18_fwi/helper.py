import math
from pathlib import Path

import cupy as cp
import numpy as np
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import point_source
from cuwave.wave import stable_dt

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

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
MIN_INDICATOR = 1e-3
LENGTH = 0.04  # the scan's extent along x, y following from its aspect ratio


# ------------------------------------- load data -------------------------------------
def load_indicator() -> np.ndarray:
    """CT slice as a density-scaling field, its void set to MIN_INDICATOR.

    The first 50 x 50 corner hosts the source and the outermost rings carry the
    boundary, so both are filled with solid before the void is floored.
    """
    indicator = np.ascontiguousarray(np.load(DATA_DIR / "B_Hai_1.npy").T)
    indicator[:50, :50] = 1.0
    indicator[:5, :] = 1.0
    indicator[-6:, :] = 1.0
    indicator[:, :5] = 1.0
    indicator[:, -6:] = 1.0
    indicator[indicator == 0] = MIN_INDICATOR
    return indicator


# --------------------------------------- setup ---------------------------------------
indicator = load_indicator()
Nx, Ny = indicator.shape

Lx, Ly = LENGTH, LENGTH * Ny / Nx
dx = (Lx / (Nx - 3), Ly / (Ny - 3))
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
frequency = WAVESPEED / (POINTS_PER_WAVELENGTH * dx[0])


def build(T: float) -> tuple[ScalarWave, object, cp.ndarray]:
    """simulation, corner point source, and padded material for a run of length T.

    Only the number of steps depends on T, so the grid above is shared and this
    is all a driver has to rebuild when it changes the simulated duration.
    """
    N = math.ceil(T / dt)
    sim = ScalarWave(
        (Nx, Ny),
        dx,
        N,
        dt,
        THREADS,
        space_order=SPACE_ORDER,
        wavespeed=WAVESPEED,
        density=DENSITY,
    )
    t = np.linspace(0, (N - 1) * dt, N)
    source = point_source(sim, (0.0, 0.0), sineburst(t, AMPLITUDE, frequency, CYCLES))

    indicator_padded = cp.ones(sim.Nx_padded, dtype=sim.dtype)
    indicator_padded[:Nx, :Ny] = cp.asarray(indicator, dtype=sim.dtype)
    return sim, source, indicator_padded
