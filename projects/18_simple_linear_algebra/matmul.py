import time
from pathlib import Path

import cupy as cp

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
N = 4096
THREADS = 16  # per block dimension

# --------------------------------------- setup ---------------------------------------
matmul = cp.RawKernel((BASE_DIR / "matmul.cu").read_text(), "matmul")
A = cp.random.rand(N, N, dtype=cp.float32)
B = cp.random.rand(N, N, dtype=cp.float32)
C = cp.empty_like(A)

threads = (THREADS, THREADS)
blocks = ((N + THREADS - 1) // THREADS, (N + THREADS - 1) // THREADS)

# --------------------------------------- solve ---------------------------------------
matmul(blocks, threads, (A, B, C, N))  # warm-up: first launch compiles kernel
cp.cuda.Stream.null.synchronize()  # wait for creating A, B and warm-up launch
tic = time.time()
matmul(blocks, threads, (A, B, C, N))
cp.cuda.Stream.null.synchronize()  # wait for kernel to finish
toc = time.time()
print(f"elapsed time {(toc - tic) * 1e3:.2f} ms with matmul.cu")

C_reference = cp.matmul(A, B)  # warm-up: first call initializes cuBLAS
cp.cuda.Stream.null.synchronize()  # wait for warm-up call
tic = time.time()
C_reference = cp.matmul(A, B)
cp.cuda.Stream.null.synchronize()  # wait for cuBLAS to finish
toc = time.time()
print(f"elapsed time {(toc - tic) * 1e3:.2f} ms with cp.matmul")

# ----------------------------------- postprocessing ----------------------------------
error = cp.linalg.norm(C - C_reference) / cp.linalg.norm(C_reference)
print(f"relative error {float(error):.2e}")
