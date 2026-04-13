import math
import random

import matplotlib.pyplot as plt
import numpy as np
from signal_config import T_base, connections, dt, sample_problem
from signal_helper import get_spiking_f
from tqdm import tqdm

from postprocessing import save_csv
from solvers.dynamic_mdof import MDOF

np.random.seed(10)
random.seed(16)
# ------------------------ data sampling settings ------------------------
T = T_base * 4
N = int(math.ceil(T / dt))
samples = (16, 16)
data = np.zeros((samples[0], samples[1], 3, N))

# ----------------------- base problem definition ------------------------
m, k, d, f_base, u0, du0 = sample_problem()

# ------------------------------- anomaly --------------------------------
bump_center = 11.4  # 7
f_index = 2
widths = np.linspace(0.1, 1, samples[0])
# widths = np.linspace(0.5, 5, samples[0]) # more challenging
heights = np.linspace(1, 10, samples[1])
widths, heights = np.meshgrid(widths, heights, indexing="ij")

for i in tqdm(range(samples[0])):
    for j in range(samples[1]):
        width, height = widths[i, j], heights[i, j]
        f = get_spiking_f(bump_center, width, height, f_base, f_index)

# -------------------------------- solve ---------------------------------
        solver = MDOF(m, k, d, f, connections)
        t, U = solver.solve(u0, du0, T, dt=dt)

# ------------------------------ save data -------------------------------
        data[i, j] = U.T

# ----------------------------- export data ------------------------------
np.savez(f"../../data/anomaly_3dof_amp.npz", X=data, widths=widths, heights=heights)

# ---------------------------- postprocessing ----------------------------
i, j = 10, 4
width, height = widths[i, j], heights[i, j]
f = get_spiking_f(bump_center, width, height, f_base, f_index)
f_ = [f(t_)[f_index] for t_ in t]

fig, ax = plt.subplots()
ax.plot(t, data[i, j, 0], "k")
ax.plot(t, data[i, j, 1], "r")
ax.plot(t, data[i, j, 2], "b")
ax1 = ax.twinx()
ax1.plot(t, f_, "gray")
plt.show()

# ----------------------------- book export ------------------------------
# second quadrant
save_csv(
    "../../results/signal_f.csv",
    x=t[: N // 4],
    y1=data[i, j, 0, N // 4 : N // 2],
    y2=data[i, j, 1, N // 4 : N // 2],
    y3=data[i, j, 2, N // 4 : N // 2],
    f=f_[N // 4 : N // 2],
)
