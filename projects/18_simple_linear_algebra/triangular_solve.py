import time

import cupy as cp
import cupyx.scipy.sparse as csparse
import cupyx.scipy.sparse.linalg as csparse_linalg
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

# -------------------------------------- settings -------------------------------------
SIZES = [10**4, 10**5, 10**6]
REPEATS = 5  # best of, shared GPUs swing with load


# --------------------------------------- helper --------------------------------------
def best_time(fun):
    times = []
    for _ in range(REPEATS):
        cp.cuda.Stream.null.synchronize()
        tic = time.perf_counter()
        fun()
        cp.cuda.Stream.null.synchronize()
        toc = time.perf_counter()
        times.append(toc - tic)
    return min(times) * 1e3


# --------------------------------------- solve ---------------------------------------
for N in SIZES:
    L = sparse.diags([-np.ones(N - 1), 2 * np.ones(N)], [-1, 0], format="csr")
    b = np.ones(N)
    L_gpu = csparse.diags([-cp.ones(N - 1), 2 * cp.ones(N)], [-1, 0], format="csr")
    b_gpu = cp.ones(N)

    y = sparse_linalg.spsolve_triangular(L, b, lower=True)
    y_gpu = csparse_linalg.spsolve_triangular(L_gpu, b_gpu, lower=True)
    assert np.allclose(y, y_gpu.get())

    multiply_cpu = best_time(lambda: L @ b)
    multiply_gpu = best_time(lambda: L_gpu @ b_gpu)
    solve_cpu = best_time(lambda: sparse_linalg.spsolve_triangular(L, b, lower=True))
    solve_gpu = best_time(lambda: csparse_linalg.spsolve_triangular(L_gpu, b_gpu, lower=True))
    print(f"N = {N:>8}: multiply cpu {multiply_cpu:8.2f} ms, gpu {multiply_gpu:8.2f} ms | solve cpu {solve_cpu:8.2f} ms, gpu {solve_gpu:8.2f} ms")
