import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from solvers.bouncing_balls import BouncingBalls

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

np.random.seed(0)

# -------------------------------------- settings -------------------------------------
# discretization
DT = 5e-3
N = 800

# physics
BOUNDS = [[0, 1], [0, 1]]  # x, y
E = 0.9
G = np.array([0, -9.81])
BALLS = 8

p0 = np.random.uniform(0.1, 0.9, (BALLS, 2))
v0 = np.random.uniform(-1, 1, (BALLS, 2))
r = np.random.uniform(0.01, 0.05, BALLS)


# ------------------------------------- simulation ------------------------------------
solver = BouncingBalls(p0, v0, r, BOUNDS, e=E, g=G)
p = solver.solve(DT, N)

# ----------------------------------- postprocessing ----------------------------------
colors = plt.cm.tab20

if not args.animate:
    fig, ax = plt.subplots(dpi=150)
    for j in range(0, N, 10):
        for i in range(BALLS):
            circle = Circle(
                p[j, i], r[i], fill=True, alpha=0.1, color=colors(i % colors.N)
            )
            ax.add_patch(circle)

    for i in range(BALLS):
        ax.plot(p[:, i, 0], p[:, i, 1], color=colors(i % colors.N))

    xmin, xmax = BOUNDS[0]
    ymin, ymax = BOUNDS[1]
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1)
    ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()

# ------------------------------ animation postprocessing -----------------------------
if args.animate:
    PLOT_EVERY = 2
    (ANIMATION_DIR / "balls").mkdir(parents=True, exist_ok=True)
    for j in range(0, N, PLOT_EVERY):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        for i in range(BALLS):
            circle = Circle(
                p[j, i], r[i], fill=True, alpha=0.9, color=colors(i % colors.N)
            )
            ax.add_patch(circle)

        xmin, xmax = BOUNDS[0]
        ymin, ymax = BOUNDS[1]
        rect = Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=4
        )
        ax.add_patch(rect)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"balls/frame_{j // PLOT_EVERY}.jpg")
        plt.close()
