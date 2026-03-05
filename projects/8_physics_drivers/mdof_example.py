from solvers.dynamic_mdof import MDOF
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- problem definition --------------------------
T = 20
dt = 0.05

m = np.array([1, 1, 1])
k = [2, 2, 2]
# k = lambda t : [2, 2, np.exp(-1e-1 * t) * 2]
d = [0.1, 0.1, 0.1]
connections = [[None, 0], [0, 1], [1, 2]]

# force
amp, freq = 1., 2.
f = lambda t : [0 * t, amp * np.sin(freq * 2 * np.pi * t), 0 * t]

# initial conditions
u0 = [0., 0., 0.]
du0 = [0., 0., 0.]

# -------------------------------- solve ---------------------------------
solver = MDOF(m, k, d, f, connections)
t, U = solver.solve(u0, du0, T, dt=dt)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(t, U[:,0], 'k')
ax.plot(t, U[:,1], 'r')
ax.plot(t, U[:,2], 'b')
plt.show()
