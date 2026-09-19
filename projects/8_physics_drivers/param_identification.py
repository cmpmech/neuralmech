import time

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------- settings -------------------------------------
SAMPLES = 64  # 32
L = 1
REGULARIZATION = 1e-3

# case 1
# u_fun = lambda x: 1 - (2 * x - 1) ** 2  # solution
# p_fun = lambda x: 16 * x + 4  # load
# EA_fun = lambda x: 1 + x  # axial stiffness

# case 2 from https://link.springer.com/book/10.1007/978-3-031-89529-6
u_fun = lambda x: np.sin(2 * np.pi * x)
p_fun = lambda x: (
    -2 * (3 * x**2 - 2 * x) * np.pi * np.cos(2 * np.pi * x)
    + 4 * (x**3 - x**2 + 1) * np.pi**2 * np.sin(2 * np.pi * x)
)
EA_fun = lambda x: x**3 - x**2 + 1

# -------------------------------------- sampling -------------------------------------
x = np.linspace(0, L, SAMPLES)
um = u_fun(x)
p = p_fun(x)


# ----------------------------- setup system of equations -----------------------------
# (EA u')' + p = 0 -> u' EA' + u'' EA = - p
tic = time.time()

# finite difference operators
h = x[1] - x[0]
D1 = np.zeros((SAMPLES, SAMPLES))  # first derivative
D2 = np.zeros((SAMPLES, SAMPLES))  # second derivative
for i in range(1, SAMPLES - 1):
    D1[i, [i - 1, i + 1]] = [-1 / (2 * h), 1 / (2 * h)]
    D2[i, [i - 1, i, i + 1]] = [1 / h**2, -2 / h**2, 1 / h**2]
D1[0, 0:3] = [-3 / (2 * h), 4 / (2 * h), -1 / (2 * h)]  # boundary
D1[-1, -3:] = [1 / (2 * h), -4 / (2 * h), 3 / (2 * h)]  # boundary
D2[0, 0:4] = [2 / h**2, -5 / h**2, 4 / h**2, -1 / h**2]  # boundary
D2[-1, -4:] = [-1 / h**2, 4 / h**2, -5 / h**2, 2 / h**2]  # boundary


dudx = D1 @ um
ddudxx = D2 @ um

A = np.diag(dudx) @ D1 + np.diag(ddudxx)  # operator acting on the unknown EA
b = -p
# --------------------------------------- solve ---------------------------------------
A_aug = np.vstack([A, np.sqrt(REGULARIZATION) * D2])  # D1 or D2
b_aug = np.concatenate([b, np.zeros(SAMPLES)])
EA = np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]

# without regularization
# EA = np.linalg.solve(A, b)
# EA = np.linalg.lstsq(A, b, rcond=None)[0]

toc = time.time()
print(f"elapsed time {1000 * (toc - tic):.2f} ms")

# ----------------------------------- postprocessing ----------------------------------
x_test = np.linspace(0, L, 100)
EA_test = EA_fun(x_test)

fig, ax = plt.subplots()
ax.plot(x_test, EA_test, "k")
ax.plot(x, EA, "ro-")
plt.show()
