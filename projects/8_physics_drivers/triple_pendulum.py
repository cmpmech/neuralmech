import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from solvers.multibody import PlanarMultibody

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# discretization
DT = 1e-2
T = 12.0

# physics
BODIES = 3
L = 1.0  # link length
MASS = 1.0
G = 9.81

# initial state
THETA0 = [2.5, 2.5, 2.5]  # absolute link angles from the x axis

# ------------------------------------- simulation ------------------------------------
bodies = []
for i in range(BODIES):
    bodies.append(
        {
            "parent": None if i == 0 else i - 1,
            "joint": "revolute",
            "anchor": (0.0, 0.0) if i == 0 else (L, 0.0),
            "com": (L, 0.0),
            "mass": MASS,
            "inertia": 0.0,  # point mass at the link tip
        }
    )

solver = PlanarMultibody(bodies, g=G)
t, q = solver.solve(THETA0, [0.0] * BODIES, T, DT)

# centers of mass over time, plus the fixed pivot at the origin for the linkage
coms = np.array([solver.forward_kinematics(q[j])[1] for j in range(len(t))])
chain = np.concatenate([np.zeros((len(t), 1, 2)), coms], axis=1)

# ----------------------------------- postprocessing ----------------------------------
if not args.animate and not args.book:
    fig, ax = plt.subplots(dpi=150)
    ax.plot(coms[:, -1, 0], coms[:, -1, 1], color="r", linewidth=0.8)
    ax.plot(chain[-1, :, 0], chain[-1, :, 1], color="k", alpha=1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

# -------------------------------- book postprocessing --------------------------------
if args.book:
    xlims = [-2.4, 2.4]  # manual
    ylims = [-3, 1.8]  # manual

    fig, ax = plt.subplots(figsize=(2, 2), dpi=250)
    ax.plot(coms[:, -3, 0], coms[:, -3, 1], color="b", linewidth=0.5, alpha=0.2)
    ax.plot(coms[:, -2, 0], coms[:, -2, 1], color="r", linewidth=0.5, alpha=0.2)
    ax.plot(coms[:, -1, 0], coms[:, -1, 1], color="k", linewidth=0.5, alpha=0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / "triple_pendulum.png", transparent=True)
    plt.close()

    print(
        (chain[-1, :, 0] - xlims[0]) / (xlims[1] - xlims[0]),
        (chain[-1, :, 1] - ylims[0]) / (ylims[1] - ylims[0]),
    )


# ------------------------------ animation postprocessing -----------------------------
if args.animate:
    PLOT_EVERY = 2
    REACH = 1.1 * BODIES * L
    (ANIMATION_DIR / "triple_pendulum").mkdir(parents=True, exist_ok=True)
    for j in range(0, len(t), PLOT_EVERY):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.plot(
            coms[: j + 1, -1, 0],
            coms[: j + 1, -1, 1],
            color="r",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.plot(chain[j, :, 0], chain[j, :, 1], color="k", linewidth=2)
        for i in range(BODIES):
            ax.add_patch(Circle(coms[j, i], 0.04 * L, color="k"))
        ax.set_xlim(-REACH, REACH)
        ax.set_ylim(-REACH, REACH)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"triple_pendulum/frame_{j // PLOT_EVERY}.jpg")
        plt.close()
