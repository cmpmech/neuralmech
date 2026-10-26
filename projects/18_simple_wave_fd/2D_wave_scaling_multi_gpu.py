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
STEPS_PER_GRAPH = 20  # even, so the graph ends with u0 and u1 in place
DEVICES = list(range(cp.cuda.runtime.getDeviceCount()))

# physics
LENGTHS = [1.0, 1.0]
WAVESPEED = 1.0

# discretization
RESOLUTIONS = np.unique(np.logspace(0.5, 5.5, 62).astype(np.int64))
STEPS = 2000
SAFETY = 0.95  # fraction of the stable time step
MEMORY = 0.9  # fraction of the free GPU memory the two fields may fill

# initial condition
CENTER = [0.5, 0.5]
SIGMA = 0.02

# --------------------------------------- setup ---------------------------------------
source = (BASE_DIR / "step2D_wave.cu").read_text()
fd_kernel = cp.RawKernel(source, "padded_step", COMPILER_OPTIONS)
name = cp.cuda.runtime.getDeviceProperties(DEVICES[0])["name"].decode()


def copy_rows(Us, lower, src, dst, start, stop):
    nbytes = (stop - start) * Us[src].shape[2] * Us[src].itemsize
    for level in range(2):
        cp.cuda.runtime.memcpyPeer(
            Us[dst][level, start - lower[dst]].data.ptr, DEVICES[dst],
            Us[src][level, start - lower[src]].data.ptr, DEVICES[src],
            nbytes,
        )


def synchronize():
    for device in DEVICES:
        cp.cuda.Device(device).synchronize()


# ---------------------------------- parametric study ---------------------------------
dofs = []
timings = []

for resolution in tqdm(RESOLUTIONS, desc=f"{len(DEVICES)}x"):
    NX = NY = int(resolution)
    dx, dy = LENGTHS[0] / (NX - 1), LENGTHS[1] / (NY - 1)
    dt = SAFETY * min(dx, dy) / WAVESPEED / math.sqrt(2)

    NY_PADDED = ((NY + 32 - 1) // 32) * 32
    bounds = np.linspace(1, NX - 1, len(DEVICES) + 1).astype(int)
    lower = np.maximum(bounds[:-1] - STEPS_PER_GRAPH, 0)
    upper = np.minimum(bounds[1:] + STEPS_PER_GRAPH, NX)
    fields = 2 * int((upper - lower).max()) * NY_PADDED * DTYPE().itemsize
    if fields > MEMORY * min(cp.cuda.Device(d).mem_info[0] for d in DEVICES):
        break

    Us, streams, graphs = [], [], []
    for g, device in enumerate(DEVICES):
        with cp.cuda.Device(device):
            rows = int(upper[g] - lower[g])
            U = cp.zeros((2, rows, NY_PADDED), dtype=DTYPE)
            x = cp.arange(lower[g], upper[g], dtype=DTYPE)[:, None] * dx
            y = cp.linspace(0, LENGTHS[1], NY, dtype=DTYPE)[None, :]
            gaussian_x = cp.exp(-((x - CENTER[0]) ** 2) / (2 * SIGMA**2))
            gaussian_y = cp.exp(-((y - CENTER[1]) ** 2) / (2 * SIGMA**2))
            cp.multiply(gaussian_x, gaussian_y, out=U[0, :, :NY])
            U[1] = U[0]
            Us.append(U)

            stream = cp.cuda.Stream(non_blocking=True)
            u0, u1 = U
            with stream:
                stream.begin_capture()
                for t in range(STEPS_PER_GRAPH):
                    fd_kernel(
                        ((NY_PADDED + THREADS_J - 1) // THREADS_J,
                         (rows + THREADS_I - 1) // THREADS_I),
                        (THREADS_J, THREADS_I),
                        (
                            u0, u1, u0,
                            DTYPE(WAVESPEED), DTYPE(dt), DTYPE(dx), DTYPE(dy),
                            rows, NY, NY_PADDED,
                        ),
                    )
                    u1, u0 = u0, u1
                graphs.append(stream.end_capture())
            graphs[g].launch(stream)  # warm-up, the first replay uploads the graph
            streams.append(stream)

    synchronize()
    tic = time.time()
    for t in range(STEPS // STEPS_PER_GRAPH):
        for g, device in enumerate(DEVICES):
            with cp.cuda.Device(device):
                graphs[g].launch(streams[g])
        synchronize()
        # each halo row comes from the slab that owns it, also when slabs are thin
        for dst in range(len(DEVICES)):
            for src in range(len(DEVICES)):
                start = max(lower[dst], bounds[src])
                stop = min(upper[dst], bounds[src + 1])
                if src != dst and start < stop:
                    copy_rows(Us, lower, src, dst, start, stop)
        synchronize()
    toc = time.time()

    timings.append((toc - tic) / STEPS)
    dofs.append((NX - 2) * (NY - 2))
    del U, Us, u0, u1, streams, graphs
    for device in DEVICES:
        with cp.cuda.Device(device):
            cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------- export --------------------------------------
save_csv(CSV_DIR / f"wave_scaling_multi_{len(DEVICES)}.csv", x=dofs, y=timings)

# ----------------------------------- postprocessing ----------------------------------
throughput = np.array(dofs) / np.array(timings)
print(f"{len(DEVICES)}x {name}")
print(f"peak {throughput.max() / 1e9:.2f} billion dofs/s")
print(f"largest grid {max(dofs) / 1e6:.1f} million dofs")

fig, ax = plt.subplots()
ax.plot(dofs, throughput, "k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("dofs")
ax.set_ylabel("dofs/s")
plt.show()
