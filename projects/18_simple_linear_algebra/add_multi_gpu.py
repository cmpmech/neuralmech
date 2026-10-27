import time
from pathlib import Path

import cupy as cp
import numpy as np

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
N = int(1e8)
THREADS = 1024
DEVICES = list(range(cp.cuda.runtime.getDeviceCount()))  # [0, 0] splits on one GPU

# ------------------------------------ prepare data -----------------------------------
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, N).astype(np.float32)
y = rng.uniform(-1, 1, N).astype(np.float32)

bounds = np.linspace(0, N, len(DEVICES) + 1).astype(int)
xs, ys, zs = [], [], []
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        xs.append(cp.asarray(x[bounds[g] : bounds[g + 1]]))
        ys.append(cp.asarray(y[bounds[g] : bounds[g + 1]]))
        zs.append(cp.empty_like(xs[g]))

# --------------------------------------- setup ---------------------------------------
add_kernel = cp.RawKernel((BASE_DIR / "add.cu").read_text(), "addParallel")


def add_multi_gpu():
    for g, device in enumerate(DEVICES):
        with cp.cuda.Device(device):
            n = len(xs[g])
            blocks = ((n + THREADS - 1) // THREADS,)
            add_kernel(blocks, (THREADS,), (xs[g], ys[g], zs[g], n))


def synchronize():
    for device in DEVICES:
        cp.cuda.Device(device).synchronize()


# --------------------------------------- solve ---------------------------------------
add_multi_gpu()  # warm-up: the first launch compiles the kernel on every GPU
synchronize()
tic = time.time()
add_multi_gpu()
synchronize()
toc = time.time()
print(f"elapsed time {(toc - tic) * 1e3:.2f} ms on {len(DEVICES)} GPUs")

# ----------------------------------- postprocessing ----------------------------------
z = np.concatenate([z_g.get() for z_g in zs])
print(f"max error {np.abs(z - (x + y)).max():.2e}")
