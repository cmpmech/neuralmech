import math
import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32
THREADS_J, THREADS_I = 128, 4
COMPILER_OPTIONS = ("--use_fast_math",)
STEPS_PER_GRAPH = 20  # even, so the graph ends with u0 and u1 in place
DEVICES = list(range(cp.cuda.runtime.getDeviceCount()))  # [0, 0] splits on one GPU

# physics
LENGTHS = [1.0, 2.0]
WAVESPEED = 1.0
T = 1.0

# discretization
NX, NY = 1600, 3200
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# ------------------------------------ prepare data -----------------------------------
dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)
N = math.ceil(T / dt / STEPS_PER_GRAPH) * STEPS_PER_GRAPH

NY_PADDED = ((NY + 32 - 1) // 32) * 32

x = np.linspace(0, LENGTHS[0], NX)
y = np.linspace(0, LENGTHS[1], NY)
x, y = np.meshgrid(x, y, indexing="ij")
gaussian = np.exp(-((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / (2 * SIGMA**2))

# each step invalidates one halo row per side, so a graph needs one halo row per step
bounds = np.linspace(1, NX - 1, len(DEVICES) + 1).astype(int)
lower = np.maximum(bounds[:-1] - STEPS_PER_GRAPH, 0)
upper = np.minimum(bounds[1:] + STEPS_PER_GRAPH, NX)
Us = []
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        U = cp.zeros((2, upper[g] - lower[g], NY_PADDED), dtype=DTYPE)
        U[:, :, :NY] = cp.asarray(gaussian[lower[g] : upper[g]])
        Us.append(U)

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR / "step2D_wave.cu").read_text()
fd_kernel = cp.RawKernel(source, "padded_step", COMPILER_OPTIONS)

blocks_j = (NY_PADDED + THREADS_J - 1) // THREADS_J


def fd_step(u0, u1, u2):
    rows = u1.shape[0]
    fd_kernel(
        (blocks_j, (rows + THREADS_I - 1) // THREADS_I),
        (THREADS_J, THREADS_I),
        (
            u0, u1, u2,
            DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy),
            rows, NY, NY_PADDED,
        ),
    )
    return u2


def copy_rows(src, dst, start, stop):
    nbytes = (stop - start) * NY_PADDED * Us[src].itemsize
    for level in range(2):
        cp.cuda.runtime.memcpyPeer(
            Us[dst][level, start - lower[dst]].data.ptr, DEVICES[dst],
            Us[src][level, start - lower[src]].data.ptr, DEVICES[src],
            nbytes,
        )


def exchange_halos():
    # each halo row comes from the slab that owns it, also when slabs are thin
    for dst in range(len(DEVICES)):
        for src in range(len(DEVICES)):
            start = max(lower[dst], bounds[src])
            stop = min(upper[dst], bounds[src + 1])
            if src != dst and start < stop:
                copy_rows(src, dst, start, stop)


def synchronize():
    for device in DEVICES:
        cp.cuda.Device(device).synchronize()


streams, graphs = [], []
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        stream = cp.cuda.Stream(non_blocking=True)
        u0, u1 = Us[g]
        with stream:
            stream.begin_capture()
            for t in range(STEPS_PER_GRAPH):
                u0 = fd_step(u0, u1, u0)
                u1, u0 = u0, u1
            graphs.append(stream.end_capture())
        streams.append(stream)

# --------------------------------------- solve ---------------------------------------
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        graphs[g].upload(streams[g])  # otherwise the first replay uploads the graph
synchronize()
tic = time.time()
for t in range(N // STEPS_PER_GRAPH):
    for g, device in enumerate(DEVICES):
        with cp.cuda.Device(device):
            graphs[g].launch(streams[g])
    synchronize()
    exchange_halos()
    synchronize()
toc = time.time()

dofs = (NX - 2) * (NY - 2)
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / N) / 1e9:.2f} billion dofs/s on {len(DEVICES)} GPUs")

# ----------------------------------- postprocessing ----------------------------------
u = np.concatenate(
    [
        U[1, bounds[g] - lower[g] : bounds[g + 1] - lower[g], :NY].get()
        for g, U in enumerate(Us)
    ]
)

scale = 0.1
fig, ax = plt.subplots(figsize=(3, 6), dpi=100)
ax.pcolormesh(x[1:-1], y[1:-1], u, cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
