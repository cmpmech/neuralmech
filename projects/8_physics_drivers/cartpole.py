import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from solvers.multibody import PlanarMultibody

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# discretization
DT = 1e-2
T = 6.0

# physics
L = 1.0  # pole length
M_CART = 1.0
M_POLE = 0.2
G = 9.81

# initial state
X0 = 0.0
THETA0 = np.pi / 2 - 0.1  # pole angle from the x axis: pi/2 is straight up

# ------------------------------------- simulation ------------------------------------
bodies = [
    {
        "parent": None,
        "joint": "prismatic",
        "anchor": (0.0, 0.0),
        "axis": (1.0, 0.0),
        "com": (0.0, 0.0),
        "mass": M_CART,
        "inertia": 0.0,
    },
    {
        "parent": 0,
        "joint": "revolute",
        "anchor": (0.0, 0.0),
        "com": (L, 0.0),
        "mass": M_POLE,
        "inertia": 0.0,  # point mass at the pole tip
    },
]

solver = PlanarMultibody(bodies, g=G)
t, q = solver.solve([X0, THETA0], [0.0, 0.0], T, DT)

# cart pivots and pole tips over time
coms = np.array([solver.forward_kinematics(q[j])[1] for j in range(len(t))])
pivots, bobs = coms[:, 0], coms[:, 1]

# ----------------------------------- postprocessing ----------------------------------
if not args.animate:
    fig, ax = plt.subplots(dpi=150)
    ax.plot(
        [pivots[:, 0].min() - L, pivots[:, 0].max() + L], [0, 0], color="k", linewidth=1
    )
    for j in range(0, len(t), 10):
        ax.plot(
            [pivots[j, 0], bobs[j, 0]], [pivots[j, 1], bobs[j, 1]], color="k", alpha=0.1
        )
    ax.plot(bobs[:, 0], bobs[:, 1], color="r", linewidth=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

# ------------------------------ animation postprocessing -----------------------------
if args.animate:
    PLOT_EVERY = 2
    CART_W, CART_H = 0.3, 0.18
    XPAD = L + CART_W
    XMIN, XMAX = pivots[:, 0].min() - XPAD, pivots[:, 0].max() + XPAD
    (ANIMATION_DIR / "cartpole").mkdir(parents=True, exist_ok=True)
    for j in range(0, len(t), PLOT_EVERY):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.plot(bobs[: j + 1, 0], bobs[: j + 1, 1], color="r", linewidth=0.8, alpha=0.6)
        ax.plot([XMIN, XMAX], [0, 0], color="k", linewidth=1)
        cart = Rectangle(
            (pivots[j, 0] - CART_W / 2, -CART_H / 2), CART_W, CART_H, color="k"
        )
        ax.add_patch(cart)
        ax.plot(
            [pivots[j, 0], bobs[j, 0]],
            [pivots[j, 1], bobs[j, 1]],
            color="k",
            linewidth=2,
        )
        ax.add_patch(Circle(bobs[j], 0.05 * L, color="k"))
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(-L - 0.3, L + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"cartpole/frame_{j // PLOT_EVERY}.jpg")
        plt.close()
