import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np

from fft_homogenization import FusedHomogenization, Homogenization, fibers, lame
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
CSV_DIR = (BASE_DIR / "../../results/data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--max-n", type=int, default=4096)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
E_FIBER, E_MATRIX, NU = 20.0, 1.0, 0.3
RTOL = 1e-4
REPEATS = 3

# name: (setup, largest n that fits the time or the 6 GB of the laptop GPU)
RUNS = {
    "cpu": (lambda lam, mu: Homogenization(lam, mu, np, np.float64), 1024),
    "gpu": (lambda lam, mu: Homogenization(lam, mu, cp, cp.float64), 2048),
    "fused": (lambda lam, mu: FusedHomogenization(lam, mu, cp.float32), args.max_n),
}
SIZES = 2 ** np.arange(5, int(np.log2(args.max_n)) + 1)

# ---------------------------------- parametric study ---------------------------------
results = {name: ([], [], []) for name in RUNS}
for n in SIZES:
    lam, mu = lame(np.where(fibers(n) > 0, E_FIBER, E_MATRIX), NU)
    for name, (setup, max_n) in RUNS.items():
        if n > max_n:
            continue
        homogenization = setup(lam, mu)
        homogenization.cg(RTOL)  # compile and plan
        times = []
        for _ in range(REPEATS):
            cp.cuda.Device().synchronize()
            tic = time.time()
            _, iterations = homogenization.cg(RTOL)
            cp.cuda.Device().synchronize()
            times.append(time.time() - tic)
        for values, v in zip(results[name], (n * n, min(times), iterations)):
            values.append(v)
        print(f"{name}: {n}^2 pixels, {min(times):.4f} s, {iterations} iterations")
    del homogenization
    cp.get_default_memory_pool().free_all_blocks()

# -------------------------------- book postprocessing --------------------------------
if args.book:
    for name, (x, y, iterations) in results.items():
        save_csv(CSV_DIR / f"fft_scaling_{name}.csv", x=x, y=y, iterations=iterations)
