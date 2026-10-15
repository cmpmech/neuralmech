import argparse
import math
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
THREADS_J, THREADS_I = 128, 4
COMPILER_OPTIONS = ("--use_fast_math",)

# physics and discretization (same problem as the other drivers)
LENGTHS = [1.0, 2.0]
WAVESPEED = 1.0
NX, NY = 1600, 3200
SAFETY = 0.95
STEPS = 2000

# Source-level count of one stencil update (see step2D_wave.cu): two second differences
# (5 flops each), their sum, the coefficient dt^2 c^2 (3), its product with the sum, and
# 2 u1 - u0 + (...) (3). The compiler may fuse or hoist some of these, so the count is
# approximate. The traffic is the ideal one: u0 and u1 are read once and u2 is written
# once, all other stencil reads are served from cache.
FLOPS_PER_DOF = 18
# name: (kernel, bytes per entry, padded row stride, launch swaps the thread axes)
KERNELS = {
    "V1 basic": ("fd_kernelV1", 4, False, True),
    "V2 coalesced": ("fd_kernelV2", 4, False, False),
    "V3 padding": ("fd_kernelV3", 4, True, False),
    "V4 shared": ("fd_kernelV4", 4, True, False),
    "V5 mixed": ("fd_kernelV5", 2, True, False),
}

# ---------------------------------- ceilings -----------------------------------------
# memory bandwidth: device-to-device copy of a buffer far larger than the L2 cache
a = cp.zeros(2**28, dtype=cp.float32)  # 1 GiB
b = cp.empty_like(a)
cp.copyto(b, a)
cp.cuda.Stream.null.synchronize()
tic = time.time()
for _ in range(20):
    cp.copyto(b, a)
cp.cuda.Stream.null.synchronize()
bandwidth = 20 * 2 * a.nbytes / (time.time() - tic)  # read and write, in byte/s
del a, b

# peak single-precision compute: a chain of fused multiply-adds held in registers
fma = cp.RawKernel(
    r"""
extern "C" __global__ void fma_chain(float* out, const float s, const int n){
    float x = threadIdx.x, y = s;
    for(int i = 0; i < n; i++){ x = x * y + s; y = y * x + s; x = x * y + s; y = y * x + s; }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}""",
    "fma_chain",
)
n_fma, blocks, threads = 4096, 4096, 256
out = cp.empty(blocks * threads, dtype=cp.float32)
fma((blocks,), (threads,), (out, np.float32(1e-9), n_fma))
cp.cuda.Stream.null.synchronize()
tic = time.time()
fma((blocks,), (threads,), (out, np.float32(1e-9), n_fma))
cp.cuda.Stream.null.synchronize()
peak = blocks * threads * n_fma * 4 * 2 / (time.time() - tic)  # flop/s
print(f"bandwidth {bandwidth / 1e9:.0f} GB/s, peak {peak / 1e12:.1f} Tflop/s")

# ----------------------------------- kernels ------------------------------------------
source = (BASE_DIR / "step2D_wave.cu").read_text()
dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)
NY_PADDED = ((NY + 32 - 1) // 32) * 32
dofs = (NX - 2) * (NY - 2)

names, intensities, gflops = [], [], []
for name, (kernel, nbytes, padded, swapped) in KERNELS.items():
    fd_kernel = cp.RawKernel(source, kernel, COMPILER_OPTIONS)
    dtype = cp.float16 if nbytes == 2 else cp.float32
    stride = NY_PADDED if padded else NY
    U = cp.zeros((2, NX, stride), dtype=dtype)
    u0, u1 = U[0], U[1]
    u0[:] = cp.random.rand(NX, stride).astype(dtype)
    u1[:] = u0[:]

    blocks_j = (NY + THREADS_J - 1) // THREADS_J
    blocks_i = (NX + THREADS_I - 1) // THREADS_I
    grid, block = ((blocks_i, blocks_j), (THREADS_I, THREADS_J)) if swapped else (
        (blocks_j, blocks_i), (THREADS_J, THREADS_I))
    args_kernel = (np.float32(WAVESPEED), np.float32(dt), np.float32(dx), np.float32(dy), NX, NY)
    if padded:
        args_kernel += (stride,)

    for step in range(STEPS + 1):
        if step == 1:  # first step is the warmup
            cp.cuda.Stream.null.synchronize()
            tic = time.time()
        fd_kernel(grid, block, (u0, u1, u0) + args_kernel)
        u1, u0 = u0, u1
    cp.cuda.Stream.null.synchronize()
    seconds = (time.time() - tic) / STEPS

    names.append(name)
    intensities.append(FLOPS_PER_DOF / (3 * nbytes))
    gflops.append(FLOPS_PER_DOF * dofs / seconds / 1e9)
    print(f"{name}: {gflops[-1]:.0f} Gflop/s, intensity {intensities[-1]:.2f} flop/byte")

# --------------------------------------- export --------------------------------------
if args.book:
    save_csv(CSV_DIR / "wave_roofline.csv", kernel=[n.replace(" ", "_") for n in names],
             intensity=intensities, gflops=gflops)
    save_csv(CSV_DIR / "wave_roofline_ceilings.csv", bandwidth=[bandwidth / 1e9],
             peak=[peak / 1e9])  # GB/s and Gflop/s

# ----------------------------------- postprocessing ----------------------------------
x = np.logspace(-1, 3, 200)
fig, ax = plt.subplots()
ax.plot(x, np.minimum(peak / 1e9, bandwidth / 1e9 * x), "k")
ax.plot(intensities, gflops, "o")
for n, i, g in zip(names, intensities, gflops):
    ax.annotate(n, (i, g))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity in flop/byte")
ax.set_ylabel("Gflop/s")
plt.show()
