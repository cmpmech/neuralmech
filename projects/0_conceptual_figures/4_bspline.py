import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- B-spline -------------------------------------
P = np.array(
    [[0, 0], [1, 2], [2, 2], [3, 0], [4, -2.5], [5.5, -2], [6, 0]], dtype=float
)
n, degree = len(P), 2

# uniform knot vector
# not clamped
knots = np.linspace(0, 1, n + degree + 1)
# clamped
# knots = np.concatenate(
#     [np.zeros(degree), np.linspace(0, 1, n - degree + 1), np.ones(degree)]
# )
print("knotvector: ", knots)

# basis functions
t = np.linspace(0, 1, 300)
basis = BSpline(knots, np.eye(n), degree)(t)  # shape (300, n)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# B-spline curve
x = BSpline(knots, P[:, 0], degree)(t)
y = BSpline(knots, P[:, 1], degree)(t)
ax[0].plot(x, y, "k")
ax[0].plot(*P.T, "o--", color="gray")
ax[0].set_ylim(-3, 3)

# basis functions
for i in range(n):
    ax[1].plot(t, basis[:, i])
ax[1].set_ylim(-1, 1)

plt.tight_layout()
plt.show()

# -------------------------------- book postprocessing --------------------------------
if args.book:
    save_csv(CSV_DIR / "bspline.csv", x=x, y=y)
    save_csv(CSV_DIR / "bspline_controlpoints.csv", x=P.T[0], y=P.T[1])
    save_csv(
        CSV_DIR / "bspline_bases.csv",
        t=t,
        **{f"b{i}": basis[:, i] for i in range(n)},
    )
