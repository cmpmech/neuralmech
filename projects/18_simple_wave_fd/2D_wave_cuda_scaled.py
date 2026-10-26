import argparse
import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32  # the kernel arguments and the reference stay single precision
THREADS_J, THREADS_I = 128, 4
COMPILER_OPTIONS = ("--use_fast_math",)

# physics
LENGTHS = [1.0, 2.0]
WAVESPEED = 1.0
T = [1.0, 3.0]  # times of the snapshots, the first one is timed

# discretization
NX, NY = 1600, 3200
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# plots
FIELD_RANGE = 0.1
ERROR_RANGE = [1e-6, 1e-2]

# ------------------------------------ prepare data -----------------------------------
dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)
N = [math.ceil(t / dt) for t in T]

# a warp updates 64 points, i.e., two blocks of 32 points with one scale each
NX_PADDED = NX
NY_PADDED = ((NY + 64 - 1) // 64) * 64

x = np.linspace(0, LENGTHS[0], NX)
y = np.linspace(0, LENGTHS[1], NY)
x, y = np.meshgrid(x, y, indexing="ij")
gaussian = np.exp(-((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / (2 * SIGMA**2))

U = cp.zeros((2, NX_PADDED, NY_PADDED), dtype=DTYPE)  # single precision reference
U[0, :NX, :NY] = cp.asarray(gaussian)
U[1] = U[0]
H = U.astype(cp.float16)  # naive half precision

# block scaling: 16-bit integers and one scale per block of 32 points of a row
blocks = U.reshape(2, NX_PADDED, NY_PADDED // 32, 32)
S = cp.abs(blocks).max(axis=3) / 32767
Q = cp.rint(blocks / cp.where(S > 0, S, 1)[..., None]).astype(cp.int16)
Q = Q.reshape(2, NX_PADDED, NY_PADDED)

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR / "step2D_wave.cu").read_text()
module = cp.RawModule(code=source, options=COMPILER_OPTIONS)
padded_kernel = module.get_function("padded_step")
mixed_kernel = module.get_function("mixed_step")
scaled_kernel = module.get_function("scaled_step")

blocks_j = (NY_PADDED + THREADS_J - 1) // THREADS_J
pair_blocks_j = (NY_PADDED // 2 + THREADS_J - 1) // THREADS_J  # one thread per pair
blocks_i = (NX_PADDED + THREADS_I - 1) // THREADS_I
threads = (THREADS_J, THREADS_I)
constants = (DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy), NX, NY, NY_PADDED)


def single_step(u0, u1, u2):
    padded_kernel((blocks_j, blocks_i), threads, (u0, u1, u2) + constants)


def naive_step(u0, u1, u2):
    mixed_kernel((pair_blocks_j, blocks_i), threads, (u0, u1, u2) + constants)


def scaled_step(u0, u1, u2):  # a time level is the pair of integers q and scales s
    scaled_kernel((pair_blocks_j, blocks_i), threads, (*u0, *u1, *u2) + constants)


def decode(u):
    q, s = u
    return (q.reshape(NX_PADDED, -1, 32) * s[..., None]).reshape(NX_PADDED, NY_PADDED)


# --------------------------------------- solve ---------------------------------------
variants = {
    "single": (single_step, [U[0], U[1]], lambda u: u.copy()),
    "naive": (naive_step, [H[0], H[1]], lambda u: u.astype(DTYPE)),
    "scaled": (scaled_step, [(Q[0], S[0]), (Q[1], S[1])], decode),
}
snapshots = {}
timings = {}
for name, (fd_step, (u0, u1), field) in variants.items():
    snapshots[name] = [field(u1)]
    for n in range(len(N)):
        steps = N[n] - (N[n - 1] if n > 0 else 0)
        cp.cuda.Stream.null.synchronize()
        tic = time.time()
        for t in range(steps):
            fd_step(u0, u1, u0)
            u1, u0 = u0, u1
        cp.cuda.Stream.null.synchronize()
        timings.setdefault(name, (time.time() - tic) / steps)
        snapshots[name].append(field(u1))

dofs = (NX - 2) * (NY - 2)
for name in ("naive", "scaled"):
    speed = dofs / timings[name] / 1e9
    print(f"{name}: {timings[name] * 1e3:.4f} ms/step ({speed:.2f} billion dofs/s)")
    for t, u, reference in zip(T, snapshots[name][1:], snapshots["single"][1:]):
        error = float(cp.linalg.norm(u - reference) / cp.linalg.norm(reference))
        print(f"  relative error to single precision at t = {t:g}: {error:.2e}")

# ----------------------------------- postprocessing ----------------------------------
field_norm = Normalize(-FIELD_RANGE, FIELD_RANGE)
error_norm = LogNorm(*ERROR_RANGE)
panels = {"wave_scaled_0": (snapshots["scaled"][0], cmr.fusion, field_norm)}
for name in ("scaled", "naive"):
    for t, u, reference in zip(T, snapshots[name][1:], snapshots["single"][1:]):
        panels[f"wave_{name}_{t:g}"] = (u, cmr.fusion, field_norm)
        panels[f"wave_{name}_error_{t:g}"] = (cp.abs(u - reference), "hot_r", error_norm)

if not args.book:
    fig, axes = plt.subplots(1, len(panels), figsize=(2 * len(panels), 4))
    for ax, (field, cmap, norm) in zip(axes, panels.values()):
        ax.pcolormesh(x, y, field[:NX, :NY].get(), cmap=cmap, norm=norm)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, (field, cmap, norm) in panels.items():
        fig, ax = plt.subplots(figsize=(3, 6), dpi=100)
        ax.pcolormesh(x, y, field[:NX, :NY].get(), cmap=cmap, norm=norm)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf")
        plt.close(fig)
