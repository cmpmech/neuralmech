import math
import random

import matplotlib.pyplot as plt
import numpy as np
from signal_config import T_base, connections, dt, sample_problem
from tqdm import tqdm

from postprocessing import save_csv
from solvers.dynamic_mdof import MDOF

np.random.seed(5)
random.seed(10)

# ------------------------ data sampling settings ------------------------
T = T_base
N = int(math.ceil(T / dt))
samples = 2048
data = np.zeros((samples, 3, N))

# -------------------------- generate problems ---------------------------
for i in tqdm(range(samples)):
    m, k, d, f, u0, du0 = sample_problem()
    solver = MDOF(m, k, d, f, connections)
    t, U = solver.solve(u0, du0, T, dt=dt)
    data[i] = U.T

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(data[0, 0], "k")
ax.plot(data[0, 1], "r")
ax.plot(data[0, 2], "b")
plt.show()


# ----------------------------- export data ------------------------------
np.save(f"../../data/normal_3dof_{N}.npy", data)


# ----------------------------- book export ------------------------------
save_csv(
    "../../results/signal_normal.csv", x=t, y1=data[0, 0], y2=data[0, 1], y3=data[0, 2]
)
