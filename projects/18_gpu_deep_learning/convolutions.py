import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
cp.random.seed(2)
np.random.seed(2)
DTYPE = np.float32
RUNS = 5
KERNEL_SIZES = [1, 3, 5, 7]  # k
SIZES = np.logspace(1.5, 3.7, 50).astype(np.int32)  # image is N x N


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
for k in KERNEL_SIZES:
    gpu_times, cpu_times = [], []
    for N in SIZES:
        image_cpu = np.random.rand(N, N).astype(DTYPE)
        kernel_cpu = np.random.rand(k, k).astype(DTYPE)
        image_gpu, kernel_gpu = cp.asarray(image_cpu), cp.asarray(kernel_cpu)

        gpu_times.append(timed(lambda: cp_ndimage.correlate(image_gpu, kernel_gpu), sync=True))
        cpu_times.append(timed(lambda: ndimage.correlate(image_cpu, kernel_cpu)))
        print(f"k={k} N={N}: gpu {gpu_times[-1]:.2e} s, cpu {cpu_times[-1]:.2e} s")

# --------------------------------------- export --------------------------------------
    if args.book:
        save_csv(CSV_DIR / f"conv_{k}.csv", x=SIZES, y1=gpu_times, y2=cpu_times)

# ----------------------------------- postprocessing ----------------------------------
    plt.loglog(SIZES, gpu_times, label=f"GPU k={k}")
    plt.loglog(SIZES, cpu_times, "--", label=f"CPU k={k}")
plt.xlabel("N")
plt.ylabel("wall-clock time in s")
plt.legend()
plt.show()
