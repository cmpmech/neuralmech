import time

import cupy as cp
import numpy as np

# -------------------------------------- settings -------------------------------------
N = int(1e8)
DEVICES = list(range(cp.cuda.runtime.getDeviceCount()))  # [0, 0] splits on one GPU

# ------------------------------------ prepare data -----------------------------------
rng = np.random.default_rng(0)
x = rng.uniform(0, 1, N).astype(np.float32)
y = rng.uniform(0, 1, N).astype(np.float32)

bounds = np.linspace(0, N, len(DEVICES) + 1).astype(int)
xs, ys = [], []
for g, device in enumerate(DEVICES):
    with cp.cuda.Device(device):
        xs.append(cp.asarray(x[bounds[g] : bounds[g + 1]]))
        ys.append(cp.asarray(y[bounds[g] : bounds[g + 1]]))

# --------------------------------------- setup ---------------------------------------
def dot_multi_gpu():
    partial_sums = []
    for g, device in enumerate(DEVICES):
        with cp.cuda.Device(device):
            partial_sums.append(cp.dot(xs[g], ys[g]))
    return sum(s_g.get() for s_g in partial_sums)


# --------------------------------------- solve ---------------------------------------
dot_multi_gpu()  # warm-up: the first call initializes cuBLAS on every GPU
tic = time.time()
s = dot_multi_gpu()
toc = time.time()
print(f"elapsed time {(toc - tic) * 1e3:.2f} ms on {len(DEVICES)} GPUs")

# ----------------------------------- postprocessing ----------------------------------
reference = np.dot(x.astype(np.float64), y.astype(np.float64))
print(f"relative error {abs(s - reference) / reference:.2e}")
