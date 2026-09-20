import math
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32
THREADS_J, THREADS_I = 128, 4
# the single and the mixed precision kernel, run over the same grid sizes
KERNELS = {0: ("fd_kernelV3", cp.float32), 1: ("fd_kernelV5", cp.float16)}

# physics
LENGTHS = [1.0, 1.0]
WAVESPEED = 1.0

# discretization
RESOLUTIONS = np.logspace(0.5, 4.5, 50).astype(np.int32)
STEPS = 2000
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# --------------------------------------- setup ---------------------------------------
module = cp.RawModule(path=str(BASE_DIR / "step2D_wave.ptx"))

# ---------------------------------- parametric study ---------------------------------
for version, (kernel_name, storage) in KERNELS.items():
    fd_kernel = module.get_function(kernel_name)
    dofs = []
    timings = []

    for resolution in RESOLUTIONS:
        NX = NY = int(resolution)
        dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
        dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)

        NX_PADDED = NX
        NY_PADDED = ((NY + 32 - 1) // 32) * 32

        U = cp.zeros((2, NX_PADDED, NY_PADDED), dtype=storage)
        u0 = U[0]
        u1 = U[1]

        x = np.linspace(0, LENGTHS[0], NX)
        y = np.linspace(0, LENGTHS[1], NY)
        x, y = np.meshgrid(x, y, indexing="ij")
        gaussian = np.exp(
            -((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) / (2 * SIGMA**2)
        )
        u0[:NX, :NY] = cp.asarray(gaussian)
        u1[:] = u0[:]

        blocks_j = (NY_PADDED + THREADS_J - 1) // THREADS_J
        blocks_i = (NX_PADDED + THREADS_I - 1) // THREADS_I

        cp.cuda.Stream.null.synchronize()
        tic = time.time()
        for t in tqdm(range(STEPS), desc=f"{kernel_name} {NX}", ncols=90):
            fd_kernel(
                (blocks_j, blocks_i),
                (THREADS_J, THREADS_I),
                (
                    u0, u1, u0,
                    DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy),
                    NX, NY, NY_PADDED,
                ),
            )
            u1, u0 = u0, u1
        cp.cuda.Stream.null.synchronize()
        toc = time.time()

        timings.append((toc - tic) / STEPS)
        dofs.append((NX - 2) * (NY - 2))

# --------------------------------------- export --------------------------------------
    save_csv(CSV_DIR / f"wave_scaling_{version}.csv", x=dofs, y=timings)

# ----------------------------------- postprocessing ----------------------------------
    throughput = np.array(dofs) / np.array(timings)
    print(f"{kernel_name} peak {throughput.max() / 1e9:.2f} billion dofs/s")
    print(f"{kernel_name} largest grid {max(dofs) / 1e6:.1f} million dofs")

    fig, ax = plt.subplots()
    ax.plot(dofs, throughput, "k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dofs")
    ax.set_ylabel("dofs/s")
    plt.show()
