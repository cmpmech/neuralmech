import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from postprocessing import save_csv
from signal_config import CONNECTIONS, DT, T_BASE, sample_problem
from solvers.dynamic_mdof import MDOF

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

np.random.seed(5)
random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
T = T_BASE
N = int(math.ceil(T / DT))
SAMPLES = 2048

# ------------------------------------ create data ------------------------------------
data = np.zeros((SAMPLES, 3, N))

for i in tqdm(range(SAMPLES)):
    m, k, d, f, u0, du0 = sample_problem()
    solver = MDOF(m, k, d, f, CONNECTIONS)
    t, U = solver.solve(u0, du0, T, dt=DT)
    data[i] = U.T

# --------------------------------------- export --------------------------------------
np.save(DATA_DIR / f"normal_3dof_{N}.npy", data)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(t, data[0, 0], "k")
ax.plot(t, data[0, 1], "r")
ax.plot(t, data[0, 2], "b")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "signal_normal.csv",
        x=t,
        y1=data[0, 0],
        y2=data[0, 1],
        y3=data[0, 2],
    )
