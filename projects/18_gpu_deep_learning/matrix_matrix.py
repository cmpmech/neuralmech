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
DTYPE = np.float32
RUNS = 10
SIZES = np.logspace(0, 4.4, 100).astype(np.int32)  # N

# ---------------------------------- parametric study ---------------------------------
gpu_times, cpu_times = [], []
for N in SIZES:
    A_gpu = cp.random.rand(N, N, dtype=DTYPE)
    B_gpu = cp.random.rand(N, N, dtype=DTYPE)
    _ = cp.matmul(A_gpu, B_gpu)  # warmup

    # cupy dispatches to cuBLAS
    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    for _ in range(RUNS):
        _ = cp.matmul(A_gpu, B_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_times.append((time.time() - tic) / RUNS)

    # numpy dispatches to the BLAS it was built with, see np.show_config()
    A_cpu, B_cpu = cp.asnumpy(A_gpu), cp.asnumpy(B_gpu)
    tic = time.time()
    for _ in range(RUNS):
        _ = np.matmul(A_cpu, B_cpu)
    cpu_times.append((time.time() - tic) / RUNS)
    print(f"N={N}: gpu {gpu_times[-1]:.2e} s, cpu {cpu_times[-1]:.2e} s")

# --------------------------------------- export --------------------------------------
if args.book:
    save_csv(CSV_DIR / "matrix_matrix.csv", x=SIZES, y1=gpu_times, y2=cpu_times)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(SIZES, gpu_times, "k", label="GPU")
ax.plot(SIZES, cpu_times, "r", label="CPU")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("wall-clock time in s")
ax.legend()
plt.show()
