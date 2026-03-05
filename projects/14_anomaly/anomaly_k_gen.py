import numpy as np
import random
import math
from tqdm import tqdm
from bump import bump
from solvers.dynamic_mdof import MDOF

np.random.seed(10)
random.seed(15)
# ------------------------ data sampling settings ------------------------
T = 6.4 * 4
dt = 0.05
samples = (16, 16)

N = int(math.ceil(T / dt))
data = np.zeros((samples[0], samples[1], 3, N))

# ----------------------- base problem definition ------------------------
m0_ = np.array([1, 1, 1])
k0_ = np.array([2, 2, 2])
d0_ = np.array([0.1, 0.1, 0.1])
m = np.random.uniform(0.9 * m0_, 1.1 * m0_)  # vary by 10 %
k0 = np.random.uniform(0.9 * k0_, 1.1 * k0_)  # vary by 10 %
d = np.random.uniform(0.9 * d0_, 1.1 * d0_)  # vary by 10 %

connections = [[None, 0], [0, 1], [1, 2]]

f = lambda t : [0 * t, 0 * t, np.sin(2 * 2 * np.pi * (t))]
freq = random.uniform(0.5, 2)
amp = random.uniform(0.1, 1)
shift = random.uniform(0, 1) # necessary?
f = lambda t : [0 * t, 0 * t, amp * np.sin(freq * 2 * np.pi * (t + shift))]

u0 = np.random.uniform(-0.1, 0.1, 3)
du0 = np.random.uniform(-0.1, 0.1, 3)

# ------------------------------- anomaly --------------------------------
bump_center = 11.4 #7
widths = np.linspace(0.5, 5, samples[0])
heights = np.linspace(1, 0.1, samples[1])
widths, heights = np.meshgrid(widths, heights, indexing='ij')

for i in tqdm(range(samples[0])):
    for j in range(samples[1]):
        width, height = widths[i, j], heights[i, j]

        def degrade(t):
            if t < bump_center:
                return bump(t, bump_center, width, height)
            else:
                return height

        def k(t):
            k_mod = k0.copy()
            k_mod[2] *= degrade(t)
            return k_mod

# -------------------------------- solve ---------------------------------
        solver = MDOF(m, k, d, f, connections)
        t, U = solver.solve(u0, du0, T, dt=dt)

# ------------------------------ save data -------------------------------
        data[i, j] = U.T

# ----------------------------- export data ------------------------------
np.savez(f'../../data/anomaly_3dof_k.npz', X=data, widths=widths, heights=heights)

# ---------------------------- postprocessing ----------------------------
k_ = [k(t_)[2] for t_ in t]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(t, U[:,0])
ax1 = ax.twinx()
ax1.plot(t, k_, 'k')
plt.show()