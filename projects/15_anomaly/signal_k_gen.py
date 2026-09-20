import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from postprocessing import save_csv
from signal_config import CONNECTIONS, DT, T_BASE, sample_problem
from signal_helper import get_degrading_k
from solvers.dynamic_mdof import MDOF

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

np.random.seed(10)
random.seed(17)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
T = T_BASE * 4
N = int(math.ceil(T / DT))
SAMPLES = (16, 16)

# anomaly
BUMP_CENTER = 11.4
K_INDEX = 2
WIDTHS = np.linspace(0.5, 5, SAMPLES[0])
HEIGHTS = np.linspace(1, 0.1, SAMPLES[1])

# ------------------------------------ create data ------------------------------------
widths, heights = np.meshgrid(WIDTHS, HEIGHTS, indexing="ij")
m, k_base, d, f, u0, du0 = sample_problem()
data = np.zeros((SAMPLES[0], SAMPLES[1], 3, N))

for i in tqdm(range(SAMPLES[0])):
    for j in range(SAMPLES[1]):
        k = get_degrading_k(BUMP_CENTER, widths[i, j], heights[i, j], k_base, K_INDEX)
        solver = MDOF(m, k, d, f, CONNECTIONS)
        t, U = solver.solve(u0, du0, T, dt=DT)
        data[i, j] = U.T

# --------------------------------------- export --------------------------------------
np.savez(DATA_DIR / "anomaly_3dof_k.npz", X=data, widths=widths, heights=heights)

# ----------------------------------- postprocessing ----------------------------------
i, j = 10, 4
k = get_degrading_k(BUMP_CENTER, widths[i, j], heights[i, j], k_base, K_INDEX)
k_history = [k(t_)[K_INDEX] for t_ in t]

fig, ax = plt.subplots()
ax.plot(t, data[i, j, 0], "k")
ax.plot(t, data[i, j, 1], "r")
ax.plot(t, data[i, j, 2], "b")
ax1 = ax.twinx()
ax1.plot(t, k_history, "gray")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    # second quadrant, where the stiffness degrades
    save_csv(
        CSV_DIR / "signal_k.csv",
        x=t[: N // 4],
        y1=data[i, j, 0, N // 4 : N // 2],
        y2=data[i, j, 1, N // 4 : N // 2],
        y3=data[i, j, 2, N // 4 : N // 2],
        k=k_history[N // 4 : N // 2],
    )
