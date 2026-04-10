import numpy as np
import math
from tqdm import tqdm
from bump import bump
from solvers.dynamic_mdof import MDOF

# ------------------------ data sampling settings ------------------------
T = 6.4 * 4
dt = 0.05
samples = (16, 16)

N = int(math.ceil(T / dt))
data = np.zeros((samples[0], samples[1], 3, N))

# ----------------------- base problem definition ------------------------
# m = np.array([1., 1., 1.])
# k = np.array([2., 2., 2.])
# d = np.array([0.1, 0.1, 0.1])
m0_ = np.array([1, 1, 1])
k0_ = np.array([2, 2, 2])
d0_ = np.array([0.1, 0.1, 0.1])
m = np.random.uniform(0.9 * m0_, 1.1 * m0_)  # vary by 10 %
k = np.random.uniform(0.9 * k0_, 1.1 * k0_)  # vary by 10 %
d = np.random.uniform(0.9 * d0_, 1.1 * d0_)  # vary by 10 %

connections = [[None, 0], [0, 1], [1, 2]]

# u0 = [0, 0, 0]
# du0 = [0, 0, 0]
u0 = np.random.uniform(-0.1, 0.1, 3)
du0 = np.random.uniform(-0.1, 0.1, 3)

# ------------------------------- anomaly --------------------------------
bump_center = 11.4 #7
widths = np.linspace(0.1, 1, samples[0]) # todo maybe logarithmic?
# widths = np.linspace(0.5, 5, samples[0]) # more challenging
heights = np.linspace(1, 10, samples[1])
widths, heights = np.meshgrid(widths, heights, indexing='ij')

for i in tqdm(range(samples[0])):
    for j in range(samples[1]):
        width, height = widths[i, j], heights[i, j]
        amp = lambda t : bump(t, bump_center, width, height)
        f = lambda t : [0 * t, 0 * t, amp(t) * 0.5 * np.sin(2 * 2 * np.pi * t)]

# -------------------------------- solve ---------------------------------
        solver = MDOF(m, k, d, f, connections)
        t, U = solver.solve(u0, du0, T, dt=dt)

# ------------------------------ save data -------------------------------
        data[i, j] = U.T

# ----------------------------- export data ------------------------------
np.savez(f'../../data/anomaly_3dof_amp.npz', X=data, widths=widths, heights=heights)

# ---------------------------- postprocessing ----------------------------
f_ = [f(t_)[2] for t_ in t]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(t, U[:,2])
ax1 = ax.twinx()
ax1.plot(t, f_, 'k', alpha=0.4)
plt.show()