"""Plate with four stacked circular voids, inspected by a linear array on its top edge.

Shared by the total focusing method and the full waveform inversion: every element of
the array fires once while all elements listen (full matrix capture). The measurement is
simulated with a wider stencil than the inversion uses, so that the inversion does not
see its own discretization error.
"""

import math

import cupy as cp
import numpy as np
from cuwave.geometry import stacked_circles
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import Sensors, line, measure, shots
from cuwave.wave import grid_coords, stable_dt

# -------------------------------------- settings -------------------------------------
RESOLUTION = (256, 256)
LENGTHS = (1.0, 1.0)
SAFETY = 0.9

# physics
T = 2.5  # traversals of the plate
WAVESPEED, DENSITY = 1.0, 1.0
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 20.0, 2

# transducer array on the top edge
ELEMENTS = 32
ARRAY_SPAN = (0.1, 0.9)

# voids
NUM_VOIDS, VOID_MAX_RADIUS, VOID_RATIO = 4, 0.05, 0.5
VOID_YRANGE, VOID_X = (0.1, 0.5), 0.5
GAMMA_VOID = 1e-4

# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(L / (n - 3) for L, n in zip(LENGTHS, Nx))
Lx, Ly = LENGTHS
elements = line((ARRAY_SPAN[0], Ly), (ARRAY_SPAN[1], Ly), ELEMENTS)


def simulation(space_order):
    """scalar wave on the plate, its time axis and the array as sources and sensors"""
    N = math.ceil(T / (SAFETY * stable_dt(dx, WAVESPEED, space_order))) + 1
    sim = ScalarWave(Nx, dx, N, T / (N - 1), (4, 64), precision="float32",
                     space_order=space_order, wavespeed=WAVESPEED, density=DENSITY)
    t = np.linspace(0, T, N)
    return sim, t, shots(sim, elements, sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)), Sensors(sim, elements)


def voids(sim):
    """indicator on the padded grid: GAMMA_VOID inside the voids, 1 elsewhere"""
    mask = stacked_circles(grid_coords(Nx, dx, dtype=sim.dtype), NUM_VOIDS, VOID_MAX_RADIUS,
                           (VOID_YRANGE[0] * Ly, VOID_YRANGE[1] * Ly), VOID_X * Lx, axis=1,
                           ratio=VOID_RATIO, order="ascending")
    return cp.where(mask, GAMMA_VOID, 1.0).astype(sim.dtype)


def full_matrix_capture(sim, sources, sensors, indicator):
    """(elements, N, elements) traces: emitter, time step, receiver"""
    return cp.stack(measure(sim, sources, indicator, sensors))
