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
COMPILER_OPTIONS = ("--use_fast_math",)
# the single and the mixed precision kernel, run over the same grid sizes, with the
# points each thread updates
KERNELS = {0: ("padded_step", cp.float32, 1), 1: ("mixed_step", cp.float16, 2)}
TILE_I = 48  # as in temporal_step
STEPS_PER_LAUNCH = 8  # as STEPS in temporal_step
LAUNCHES_PER_GRAPH = 2  # even, so the two pairs of arrays return to their roles

# physics
LENGTHS = [1.0, 1.0]
WAVESPEED = 1.0

# discretization
RESOLUTIONS = np.unique(np.logspace(0.5, 5.0, 56).astype(np.int64))
MEMORY = 0.9  # fraction of the free GPU memory the two fields may fill
STEPS = 2000
STEPS_PER_GRAPH = 20  # even, so the graph ends with u0 and u1 in place
SAFETY = 0.95  # fraction of the stable time step

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# --------------------------------------- setup ---------------------------------------
module = cp.RawModule(code=(BASE_DIR / "step2D_wave.cu").read_text(), options=COMPILER_OPTIONS)

# ---------------------------------- parametric study ---------------------------------
for version, (kernel_name, storage, points) in KERNELS.items():
    fd_kernel = module.get_function(kernel_name)
    dofs = []
    timings = []

    for resolution in tqdm(RESOLUTIONS, desc=kernel_name):
        NX = NY = int(resolution)
        dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
        dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)

        NX_PADDED = NX
        NY_PADDED = ((NY + 32 - 1) // 32) * 32

        fields = 2 * NX_PADDED * NY_PADDED * storage().itemsize
        if fields > MEMORY * cp.cuda.Device().mem_info[0]:
            break

        U = cp.zeros((2, NX_PADDED, NY_PADDED), dtype=storage)
        u0 = U[0]
        u1 = U[1]

        # the gaussian is separable, so no full-size temporaries are needed
        x = cp.linspace(0, LENGTHS[0], NX, dtype=DTYPE)[:, None]
        y = cp.linspace(0, LENGTHS[1], NY, dtype=DTYPE)[None, :]
        gaussian_x = cp.exp(-((x - CENTER[0]) ** 2) / (2 * SIGMA**2))
        gaussian_y = cp.exp(-((y - CENTER[1]) ** 2) / (2 * SIGMA**2))
        cp.multiply(gaussian_x, gaussian_y, out=u0[:NX, :NY])
        u1[:] = u0[:]

        blocks_j = (NY_PADDED // points + THREADS_J - 1) // THREADS_J
        blocks_i = (NX_PADDED + THREADS_I - 1) // THREADS_I

        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            for t in range(STEPS_PER_GRAPH):
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
            graph = stream.end_capture()
        graph.launch(stream)  # warm-up, the first replay uploads the graph

        stream.synchronize()
        tic = time.time()
        for t in range(STEPS // STEPS_PER_GRAPH):  # no progress bar, it would be timed
            graph.launch(stream)
        stream.synchronize()
        toc = time.time()

        timings.append((toc - tic) / STEPS)
        dofs.append((NX - 2) * (NY - 2))
        del U, u0, u1, graph
        cp.get_default_memory_pool().free_all_blocks()

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

# --------------------------------- temporal blocking ---------------------------------
fd_kernel = module.get_function("temporal_step")
inner_j = THREADS_J - 2 * STEPS_PER_LAUNCH
inner_i = TILE_I - 2 * STEPS_PER_LAUNCH
steps_per_graph = LAUNCHES_PER_GRAPH * STEPS_PER_LAUNCH
dofs = []
timings = []

for resolution in tqdm(RESOLUTIONS, desc="temporal_step"):
    NX = NY = int(resolution)
    dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
    dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)

    NX_PADDED = NX
    NY_PADDED = ((NY + 32 - 1) // 32) * 32

    fields = 4 * NX_PADDED * NY_PADDED * DTYPE().itemsize  # two pairs of arrays
    if fields > MEMORY * cp.cuda.Device().mem_info[0]:
        break

    U = cp.zeros((2, NX_PADDED, NY_PADDED), dtype=DTYPE)
    V = cp.zeros_like(U)
    u0, u1 = U
    v0, v1 = V

    x = cp.linspace(0, LENGTHS[0], NX, dtype=DTYPE)[:, None]
    y = cp.linspace(0, LENGTHS[1], NY, dtype=DTYPE)[None, :]
    gaussian_x = cp.exp(-((x - CENTER[0]) ** 2) / (2 * SIGMA**2))
    gaussian_y = cp.exp(-((y - CENTER[1]) ** 2) / (2 * SIGMA**2))
    cp.multiply(gaussian_x, gaussian_y, out=u0[:NX, :NY])
    u1[:] = u0[:]

    blocks_j = (NY + inner_j - 1) // inner_j
    blocks_i = (NX + inner_i - 1) // inner_i

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for t in range(LAUNCHES_PER_GRAPH):
            fd_kernel(
                (blocks_j, blocks_i),
                (THREADS_J, THREADS_I),
                (
                    u0, u1, v0, v1,
                    DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy),
                    NX, NY, NY_PADDED,
                ),
            )
            u0, u1, v0, v1 = v0, v1, u0, u1
        graph = stream.end_capture()
    graph.launch(stream)  # warm-up, the first replay uploads the graph

    stream.synchronize()
    tic = time.time()
    for t in range(STEPS // steps_per_graph):  # no progress bar, it would be timed
        graph.launch(stream)
    stream.synchronize()
    toc = time.time()

    timings.append((toc - tic) / (STEPS // steps_per_graph * steps_per_graph))
    dofs.append((NX - 2) * (NY - 2))
    del U, V, u0, u1, v0, v1, graph
    cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------- export --------------------------------------
save_csv(CSV_DIR / "wave_scaling_temporal.csv", x=dofs, y=timings)

# ----------------------------------- postprocessing ----------------------------------
throughput = np.array(dofs) / np.array(timings)
print(f"temporal_step peak {throughput.max() / 1e9:.2f} billion dofs/s")
print(f"temporal_step largest grid {max(dofs) / 1e6:.1f} million dofs")

fig, ax = plt.subplots()
ax.plot(dofs, throughput, "k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("dofs")
ax.set_ylabel("dofs/s")
plt.show()
