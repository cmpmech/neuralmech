import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.signal import hilbert

from plate import CYCLES, FREQUENCY, Lx, Ly, WAVESPEED, elements, full_matrix_capture, simulation, voids
from postprocessing import save_csv
from tfm import tfm, tfm_kernel

BASE_DIR = Path(__file__).parent
CSV_DIR = (BASE_DIR / "../../results/data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--max-pixels", type=int, default=4096, help="largest image side")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
SPACE_ORDER = 8
REPEATS = 3
SIZES = 2 ** np.arange(5, int(np.log2(args.max_pixels)) + 1)

# ------------------------------------ measurement ------------------------------------
sim, t, sources, sensors = simulation(SPACE_ORDER)
truth = voids(sim)
scattered = full_matrix_capture(sim, sources, sensors, truth)
scattered -= full_matrix_capture(sim, sources, sensors, cp.ones_like(truth))
h = cp.ascontiguousarray(hilbert(scattered, axis=1).transpose(0, 2, 1)).astype(cp.complex64)
t0 = CYCLES / (2 * FREQUENCY)

# name: (image function of the pixel grid on the device, largest side within the time)
RUNS = {
    "cpu": (lambda x, y: tfm(h.get(), elements, t0, t[1], WAVESPEED, x.get(), y.get()), 256),
    "cupy": (lambda x, y: tfm(h, elements, t0, t[1], WAVESPEED, x, y, cp), 1024),
    "kernel": (lambda x, y: tfm_kernel(h, elements, t0, t[1], WAVESPEED, x, y), args.max_pixels),
}

# ---------------------------------- parametric study ---------------------------------
results = {name: ([], []) for name in RUNS}
for n in SIZES:
    x, y = cp.meshgrid(cp.linspace(0, Lx, n, dtype=cp.float32), cp.linspace(0, Ly, n, dtype=cp.float32), indexing="ij")
    images = {}
    for name, (run, max_n) in RUNS.items():
        if n > max_n:
            continue
        images[name] = cp.asarray(run(x, y))  # compile
        times = []
        for _ in range(REPEATS):
            cp.cuda.Device().synchronize()
            tic = time.time()
            run(x, y)
            cp.cuda.Device().synchronize()
            times.append(time.time() - tic)
        results[name][0].append(n * n)
        results[name][1].append(min(times))
    error = max(float(cp.abs(v - images["kernel"]).max() / images["kernel"].max()) for v in images.values())
    print(f"{n}^2 pixels: " + ", ".join(f"{k} {v[1][-1]:.4f} s" for k, v in results.items() if v[0][-1] == n * n)
          + f", largest difference {error:.1e}")

# -------------------------------- book postprocessing --------------------------------
if args.book:
    for name, (x, y) in results.items():
        save_csv(CSV_DIR / f"tfm_scaling_{name}.csv", x=x, y=y)
