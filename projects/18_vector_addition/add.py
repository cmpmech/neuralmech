import time
from pathlib import Path

import cupy as cp

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
PARALLEL = True  # False runs the same addition in a single thread
N = int(1e6)
THREADS = 1024

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR / "add.cu").read_text()
if PARALLEL:
    add_kernel = cp.RawKernel(source, "addParallel")
    threads = (THREADS,)
    blocks = ((N + THREADS - 1) // THREADS,)
else:
    add_kernel = cp.RawKernel(source, "addSerial")
    threads = (1,)
    blocks = (1,)

x = cp.random.uniform(-1, 1, N, dtype=cp.float32)
y = cp.random.uniform(-1, 1, N, dtype=cp.float32)
z = cp.zeros(N, dtype=cp.float32)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
add_kernel(blocks, threads, (x, y, z, N))
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {(toc - tic) * 1e3:.2f} ms")
