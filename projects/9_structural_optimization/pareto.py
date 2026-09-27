import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

np.random.seed(0)

# -------------------------------------- settings -------------------------------------
# two objectives of two variables in [0, 1]: C1 = x1 moves along the front, and
# C2 = (1 + x2) h(x1) lifts off it, so the front is C2 = h(C1); h is a steep convex drop
# plus a sigmoid step, decreasing everywhere, with a nonconvex hump in between
BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
DROP = 0.3  # height of the convex drop
DROP_WIDTH = 0.05
STEP_CENTER = 0.6  # position and width of the sigmoid step
STEP_WIDTH = 0.06

# optimization
STARTS = 5  # multistart, SLSQP is local
EPSILONS = 40  # bounds on the first objective in the epsilon-constraint sweep
WEIGHTS = 40  # weights of the weighted-sum sweep
SAMPLES = 150  # grid resolution per design variable for the objective space


# --------------------------------------- helper --------------------------------------
def sigmoid(c1):
    return 1.0 / (1.0 + np.exp((c1 - STEP_CENTER) / STEP_WIDTH))


def exact_front(c1):
    step = (sigmoid(c1) - sigmoid(1.0)) / (sigmoid(0.0) - sigmoid(1.0))
    return DROP * np.exp(-c1 / DROP_WIDTH) + (1.0 - DROP) * step


def cost1(x):
    return x[0]


def cost2(x):
    return (1.0 + x[1]) * exact_front(x[0])


def objectives(x):
    return np.array([cost1(x), cost2(x)])


def optimize(cost, constraints=()):
    """best of STARTS local SLSQP runs."""
    scipy_constraints = [{"type": "ineq", "fun": c} for c in constraints]
    best = None
    for _ in range(STARTS):
        start = np.random.uniform(*np.array(BOUNDS).T)
        result = minimize(cost, start, bounds=BOUNDS, constraints=scipy_constraints,
                          method="SLSQP")
        if result.success and all(c(result.x) >= -1e-9 for c in constraints):
            if best is None or result.fun < best.fun:
                best = result
    return best.x


def lower_hull(points):
    """lower convex hull of points sorted by their first coordinate."""
    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull = []
    for p in points:
        while len(hull) >= 2 and turn(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return np.array(hull)


# ------------------------------------ optimization -----------------------------------
# anchor points: each objective minimized alone, used to normalize the weighted sum
anchor1 = objectives(optimize(cost1))
anchor2 = objectives(optimize(cost2))
utopia = np.array([anchor1[0], anchor2[1]])
nadir = np.array([anchor2[0], anchor1[1]])

# epsilon-constraint: minimize the second objective below a bound on the first
front = []
for epsilon in np.linspace(utopia[0], nadir[0], EPSILONS):
    x = optimize(cost2, [lambda x, e=epsilon: e - cost1(x)])
    front.append(objectives(x))
front = np.array(front)

# weighted sum of the normalized objectives; alpha = 0 or 1 would drop one objective and
# admit dominated designs, so both are excluded
weighted = []
for alpha in np.linspace(0.0, 1.0, WEIGHTS + 2)[1:-1]:
    def cost(x, alpha=alpha):
        normalized = (objectives(x) - utopia) / (nadir - utopia)
        return alpha * normalized[0] + (1.0 - alpha) * normalized[1]
    weighted.append(objectives(optimize(cost)))
weighted = np.array(weighted)

# objective space: every design of a grid over the design box
x1, x2 = np.meshgrid(np.linspace(*BOUNDS[0], SAMPLES), np.linspace(*BOUNDS[1], SAMPLES))
cloud = np.array([objectives(x) for x in zip(x1.ravel(), x2.ravel())])

# ----------------------------------- postprocessing ----------------------------------
# convex hull of the exact front; its longest segment bridges the nonconvex part
c1 = np.linspace(0.0, 1.0, 20001)
hull = lower_hull(np.column_stack([c1, exact_front(c1)]))
bridge = np.argmax(np.diff(hull[:, 0]))
tangents = hull[bridge:bridge + 2]
found = np.sort(weighted[:, 0])
gap = np.argmax(np.diff(found))

print(f"monotone front: {np.all(np.diff(exact_front(c1)) <= 0)}")
print(f"epsilon-constraint deviation from the exact front "
      f"{np.abs(front[:, 1] - exact_front(front[:, 0])).max():.2e}")
print(f"convex hull bridges C1 = {tangents[0, 0]:.4f} to {tangents[1, 0]:.4f}")
print(f"weighted sum: {len(np.unique(weighted.round(6), axis=0))} distinct designs, "
      f"none between C1 = {found[gap]:.2f} and {found[gap + 1]:.2f}")

if not args.book:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(*cloud.T, s=1, color="0.85", label="designs")
    ax.plot(*front.T, "k-", label="epsilon-constraint")
    ax.plot(*tangents.T, "b--", label="convex hull")
    ax.plot(*weighted.T, "ro", mfc="none", label="weighted sum")
    ax.set_xlabel("$C_1$")
    ax.set_ylabel("$C_2$")
    ax.legend()
    plt.tight_layout()
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "pareto_front.csv", c1=front[:, 0], c2=front[:, 1])
    save_csv(CSV_DIR / "pareto_weighted.csv", c1=weighted[:, 0], c2=weighted[:, 1])
    save_csv(CSV_DIR / "pareto_hull.csv", c1=tangents[:, 0], c2=tangents[:, 1])
