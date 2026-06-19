import argparse
import math
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from solvers.wave import setup_simulation, setup_source, simulate

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")  # TODO could be animated
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DIM = 2  # 1, 2 or 3 # TODO use DIM in other files as well
FORMULATION = "scalar"  # "scalar" or "acoustic"
PRECISION = "float32"  # "float32" or "float64"
THREADS = (4, 128) if DIM == 2 else (4, 4, 64) if DIM == 3 else (128,)

# physics
LENGTH = 0.05
WAVESPEED = 6000.0
DENSITY = 2700.0
AMPLITUDE = 1e8
CYCLES = 3
T = 1e-5

# discretization (coarser in 3D to keep the toy run small)
RESOLUTION = 500 if DIM == 2 else 100 if DIM == 3 else 2000

# --------------------------------------- setup ---------------------------------------
Nx = (RESOLUTION,) * DIM
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = 0.95 * min(dx) / WAVESPEED / math.sqrt(DIM)
frequency = WAVESPEED / (20.0 * min(dx))
N = math.ceil(T / dt)

sim = setup_simulation(
    Nx,
    dx,
    N,
    dt,
    WAVESPEED,
    DENSITY,
    THREADS,
    formulation=FORMULATION,
    precision=PRECISION,
)

indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)


# --------------------------------------- helper --------------------------------------
def sineburst(t, amplitude, frequency, cycles):
    mask = (t > 0) & (t <= cycles / frequency)
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * frequency * t)
        * np.sin(np.pi * frequency * t / cycles) ** 2
    )


t_arr = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_arr, AMPLITUDE, frequency, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)

source_pos = cp.array([[n // 2] for n in Nx], dtype=cp.int32)
source = setup_source(source_pos, signal)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = u.get()
scale = float(np.max(np.abs(u_np)))

if not args.book and not args.animate:
    if DIM == 1:
        x = np.linspace(0, LENGTH, Nx[0])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, u_np, "k")
    else:
        field = u_np if DIM == 2 else u_np[Nx[0] // 2]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pcolormesh(field.T, cmap="seismic", vmin=-scale, vmax=scale)
        ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
