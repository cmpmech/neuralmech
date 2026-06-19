import matplotlib.pyplot as plt
import numpy as np

from solvers.dynamic_mdof import MDOF

# -------------------------------------- settings -------------------------------------
# discretization
T = 20
DT = 0.05

# physics
M = np.array([1, 1, 1])
K = [2, 2, 2]
# K = lambda t : [2, 2, np.exp(-1e-1 * t) * 2]
D = [0.1, 0.1, 0.1]
CONNECTIONS = [[None, 0], [0, 1], [1, 2]]
# force
AMP, FREQ = 1.0, 2.0
f = lambda t: [0 * t, AMP * np.sin(FREQ * 2 * np.pi * t), 0 * t]

# initial conditions
u0 = [0.0, 0.0, 0.0]
du0 = [0.0, 0.0, 0.0]

# --------------------------------------- solve ---------------------------------------
solver = MDOF(M, K, D, f, CONNECTIONS)
t, u = solver.solve(u0, du0, T, dt=DT)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(t, u[:, 0], "k")
ax.plot(t, u[:, 1], "r")
ax.plot(t, u[:, 2], "b")
plt.show()
