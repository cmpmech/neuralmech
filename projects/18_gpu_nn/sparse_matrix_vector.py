import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--mkl", action="store_true", help="CPU: MKL in double precision")
args = parser.parse_args()

if args.mkl:
    import sparse_dot_mkl

# -------------------------------------- settings -------------------------------------
cp.random.seed(2)
np.random.seed(2)
DTYPE = np.float32
RUNS = 5
SPARSITIES = [0.8, 0.9, 0.95, 0.99]  # s


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
for s in SPARSITIES:
    sizes = np.logspace(2, 4.7 if s <= 0.9 else 5.0, 100).astype(np.int32)  # N
    gpu_times, cpu_times = [], []
    for N in sizes:
        nnz_per_row = max(1, int((1 - s) * N))
        rows = np.repeat(np.arange(N), nnz_per_row)
        cols = np.random.randint(0, N, size=N * nnz_per_row)
        data = np.random.rand(N * nnz_per_row).astype(DTYPE)
        A_cpu = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        x_cpu = np.random.rand(N).astype(DTYPE)

        A_gpu, x_gpu = cp_sp.csr_matrix(A_cpu), cp.asarray(x_cpu)  # cuSPARSE
        gpu_times.append(timed(lambda: A_gpu @ x_gpu, sync=True))

        if args.mkl:
            A_cpu, x_cpu = A_cpu.astype(np.float64), x_cpu.astype(np.float64)
            cpu_times.append(timed(lambda: sparse_dot_mkl.dot_product_mkl(A_cpu, x_cpu)))
        else:
            cpu_times.append(timed(lambda: A_cpu @ x_cpu))
        print(f"s={s} N={N}: gpu {gpu_times[-1]:.2e} s, cpu {cpu_times[-1]:.2e} s")

# --------------------------------------- export --------------------------------------
    if args.book:
        save_csv(CSV_DIR / f"sparse_matrix_vector_{s}_{args.mkl}.csv",
                 x=sizes, y1=gpu_times, y2=cpu_times)

# ----------------------------------- postprocessing ----------------------------------
    plt.loglog(sizes, gpu_times, label=f"GPU s={s}")
    plt.loglog(sizes, cpu_times, "--", label=f"CPU s={s}")
plt.xlabel("N")
plt.ylabel("wall-clock time in s")
plt.legend()
plt.show()
