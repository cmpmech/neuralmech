import math
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32
THREADS = 1024
COMPILER_OPTIONS = ("--use_fast_math",)
DEVICES = list(range(cp.cuda.runtime.getDeviceCount()))  # [0, 0] splits on one GPU

# physics
LENGTH = 1.0
WAVESPEED = 1.0
T = 1.0

# discretization
NX = 10_000
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = 0.5
SIGMA = 0.02

# ------------------------------------ prepare data -----------------------------------
dx = LENGTH / (NX - 1)
dt = SAFETY * dx / WAVESPEED
N = math.ceil(T / dt)

x = np.linspace(0, LENGTH, NX)
gaussian = np.exp(-((x - CENTER) ** 2) / (2 * SIGMA**2))

# each GPU owns the points [bounds[g], bounds[g + 1]) plus one halo point per side
bounds = np.linspace(0, NX, len(DEVICES) + 1).astype(int)
lower = np.maximum(bounds[:-1] - 1, 0)
upper = np.minimum(bounds[1:] + 1, NX)
u0s, u1s = [], []
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        u0s.append(cp.asarray(gaussian[lower[g] : upper[g]], dtype=DTYPE))
        u1s.append(u0s[g].copy())

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR / "step1D_wave.cu").read_text()
fd_kernel = cp.RawKernel(source, "step1D", COMPILER_OPTIONS)


def fd_step(u0, u1, u2):
    n = len(u1)
    fd_kernel(
        ((n + THREADS - 1) // THREADS,),
        (THREADS,),
        (u0, u1, u2, DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), n),
    )
    return u2


def copy_point(us, src, dst, i):
    cp.cuda.runtime.memcpyPeer(
        us[dst][i - lower[dst]].data.ptr, DEVICES[dst],
        us[src][i - lower[src]].data.ptr, DEVICES[src],
        us[src].itemsize,
    )


def exchange_halos(us):
    for g in range(len(DEVICES) - 1):
        copy_point(us, g, g + 1, bounds[g + 1] - 1)  # into the left halo of g + 1
        copy_point(us, g + 1, g, bounds[g + 1])  # into the right halo of g


def synchronize():
    for device in DEVICES:
        cp.cuda.Device(device).synchronize()


# --------------------------------------- solve ---------------------------------------
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        fd_step(u0s[g], u1s[g], cp.empty_like(u0s[g]))  # warm-up: compiles the kernel
synchronize()
tic = time.time()
for t in range(N):
    for g, device in enumerate(DEVICES):
        with cp.cuda.Device(device):
            fd_step(u0s[g], u1s[g], u0s[g])
    u0s, u1s = u1s, u0s
    exchange_halos(u1s)
synchronize()
toc = time.time()

dofs = NX - 2
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")
print(f"{dofs / ((toc - tic) / N) / 1e9:.2f} billion dofs/s on {len(DEVICES)} GPUs")

# ----------------------------------- postprocessing ----------------------------------
u = np.concatenate(
    [
        u1[bounds[g] - lower[g] : bounds[g + 1] - lower[g]].get()
        for g, u1 in enumerate(u1s)
    ]
)

fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
ax.plot(x, u, "k")
plt.show()
