import argparse
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
cp.random.seed(2)
rng = np.random.default_rng(2)
DTYPE = np.float32
RUNS = 5
KERNEL_SIZES = [1, 3, 5, 7]  # K
CHANNELS = [1, 64]  # C, input and output channels of the layer
MAX_SIZE = {1: 8192, 64: 1024}  # largest image side N


# --------------------------------------- setup ---------------------------------------
def conv2d(xp, image, weight):
    C, N, _ = image.shape
    k = weight.shape[-1]
    padded = xp.pad(image, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
    windows = xp.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(1, 2))
    columns = windows.transpose(1, 2, 0, 3, 4).reshape(N * N, C * k * k)
    return columns @ weight.reshape(weight.shape[0], C * k * k).T


def timed(fun, sync=False):
    fun()  # warmup
    if sync:
        cp.cuda.Stream.null.synchronize()
    tic = time.time()
    for _ in range(RUNS):
        fun()
    if sync:
        cp.cuda.Stream.null.synchronize()
    return (time.time() - tic) / RUNS


# ---------------------------------- parametric study ---------------------------------
times = {}
sizes = {}
for C in CHANNELS:
    sizes[C] = np.unique(np.logspace(1, np.log10(MAX_SIZE[C]), 15).astype(int))
    for k in KERNEL_SIZES:
        gpu_times, cpu_times = [], []
        weight_cpu = rng.standard_normal((C, C, k, k)).astype(DTYPE)
        weight_gpu = cp.asarray(weight_cpu)
        for N in sizes[C]:
            image_cpu = rng.standard_normal((C, N, N)).astype(DTYPE)
            image_gpu = cp.asarray(image_cpu)
            gpu_times.append(timed(lambda: conv2d(cp, image_gpu, weight_gpu), sync=True))
            cpu_times.append(timed(lambda: conv2d(np, image_cpu, weight_cpu)))
        times[C, k] = gpu_times, cpu_times
        print(f"C={C} k={k}: speed-up {cpu_times[-1] / gpu_times[-1]:.2f} at N={N}")

# --------------------------------------- export --------------------------------------
if args.book:
    for C, k in times:
        save_csv(CSV_DIR / f"conv_{C}_{k}.csv", x=sizes[C], y1=times[C, k][0], y2=times[C, k][1])

# ----------------------------------- postprocessing ----------------------------------
fig, axes = plt.subplots(1, 2, sharey=True)
for ax, C in zip(axes, CHANNELS):
    for k in KERNEL_SIZES:
        gpu_times, cpu_times = times[C, k]
        ax.loglog(sizes[C], np.array(cpu_times) / np.array(gpu_times))
    ax.set_xlabel("N")
axes[0].set_ylabel("speed-up")
plt.show()
