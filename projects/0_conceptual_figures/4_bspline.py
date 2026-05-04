import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

P = np.array([[0, 0], [1, 2], [2, 2], [3, 0], [4, -2], [5, -2], [6, 0]], dtype=float)
n, degree = len(P), 2

# Clamped uniform knot vector
# knots = np.concatenate(
#     [np.zeros(degree), np.linspace(0, 1, n - degree + 1), np.ones(degree)]
# )
knots = np.linspace(0, 1, n + degree + 1)

# Basis functions: use identity matrix as control points
# t = np.linspace(0, 1, 300)
t = np.linspace(knots[degree], knots[-degree - 1], 300)
basis = BSpline(knots, np.eye(n), degree)(t)  # shape (300, n)

fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# B-spline curve
x = BSpline(knots, P[:, 0], degree)(t)
y = BSpline(knots, P[:, 1], degree)(t)
axes[0].plot(x, y, "k")
axes[0].plot(*P.T, "o--", color="gray")
axes[0].set_title("B-spline curve")

# Basis functions
for i in range(n):
    axes[1].plot(t, basis[:, i], label=f"$N_{{{i},{degree}}}$")
axes[1].legend(loc="upper right")
axes[1].set_title("Basis functions")

plt.tight_layout()
plt.show()


# # Print TikZ coordinates
# coords = " ".join(f"({xi:.4f},{yi:.4f})" for xi, yi in zip(x, y))
# print(f"\\draw[thick] plot coordinates {{{coords}}};")
