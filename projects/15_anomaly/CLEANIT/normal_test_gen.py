import numpy as np
import math
import random
from tqdm import tqdm
from solvers.dynamic_mdof import MDOF

np.random.seed(1)
random.seed(2)

# ------------------------ data sampling settings ------------------------
T = 6.4 * 4
dt = 0.05  # set with np.sqrt(np.min(m) / np.max(k)) * 0.1
samples = 128 #512

N = int(math.ceil(T / dt))
data = np.zeros((samples, 3, N))

# ----------------------- base problem definition ------------------------
m0 = np.array([1, 1, 1])
k0 = np.array([2, 2, 2])
d0 = np.array([0.1, 0.1, 0.1])

connections = [[None, 0], [0, 1], [1, 2]]

f0 = lambda freq, amp, shift: lambda t: [0 * t, 0 * t,
                     amp * np.sin(freq * 2 * np.pi * (t + shift))]

# -------------------------- generate problems ---------------------------
for i in tqdm(range(samples)):
    m = np.random.uniform(0.9 * m0, 1.1 * m0)  # vary by 10 %
    k = np.random.uniform(0.9 * k0, 1.1 * k0)  # vary by 10 %
    d = np.random.uniform(0.9 * d0, 1.1 * d0)  # vary by 10 %

    u0 = np.random.uniform(-0.1, 0.1, 3)
    du0 = np.random.uniform(-0.1, 0.1, 3)

    freq = random.uniform(0.5, 2)
    amp = random.uniform(0.1, 1)

    shift = random.uniform(0, 1)

    f = f0(freq, amp, shift)

# -------------------------------- solve ---------------------------------
    solver = MDOF(m, k, d, f, connections)
    t, U = solver.solve(u0, du0, T, dt=dt)

# ------------------------------ save data -------------------------------
    data[i] = U.T

# ----------------------------- export data ------------------------------
np.save(f'../../data/normal_3dof_test.npy', data)