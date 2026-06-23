import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solvers.dynamic_mdof import MDOF

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
T = 20
DT = 0.05

M = np.array([1, 1, 1])
K = [2, 2, 2]
D = [0.1, 0.1, 0.1]
CONNECTIONS = [[None, 0], [0, 1], [1, 2]]
AMP, FREQ = 1.0, 2.0
f = lambda t: [0 * t, AMP * np.sin(FREQ * 2 * np.pi * t), 0 * t]

u0 = [0.0, 0.0, 0.0]
du0 = [0.0, 0.0, 0.0]

# --------------------------------------- solve ---------------------------------------
solver = MDOF(M, K, D, f, CONNECTIONS)
t, u = solver.solve(u0, du0, T, dt=DT)

# ----------------------------------- postprocessing ----------------------------------
if not args.animate:
    fig, ax = plt.subplots()
    ax.plot(t, u[:, 0], "k")
    ax.plot(t, u[:, 1], "r")
    ax.plot(t, u[:, 2], "b")
    plt.show()

# ------------------------------ animation postprocessing -----------------------------
if args.animate:
    PLOT_EVERY = 1
    N_DOFS = len(M)
    REST_SPACING = 2.0
    DISPLAY_SCALE = 10.0
    X_REST = REST_SPACING * np.arange(N_DOFS, dtype=float)

    N_TEETH = 8
    SPRING_AMP = 0.1

    def draw_spring(ax, x1, x2, y0):
        lead = (x2 - x1) * 0.12
        ix1, ix2 = x1 + lead, x2 - lead
        L = ix2 - ix1
        peak_xs = [ix1 + (i + 0.5) * L / N_TEETH for i in range(N_TEETH)]
        peak_ys = [y0 + SPRING_AMP * (1 if i % 2 == 0 else -1) for i in range(N_TEETH)]
        xs = [x1, ix1] + peak_xs + [ix2, x2]
        ys = [y0, y0] + peak_ys + [y0, y0]
        ax.plot(xs, ys, color="k", linewidth=1.5)

    u_max_vis = np.max(np.abs(u)) * DISPLAY_SCALE
    x_lim = (
        X_REST[0] - REST_SPACING * 1.2,
        X_REST[-1] + u_max_vis + REST_SPACING * 0.3,
    )
    y_lim = (-1.0, 1.0)

    (ANIMATION_DIR / "mdof1d").mkdir(parents=True, exist_ok=True)

    for j in range(0, len(t), PLOT_EVERY):
        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)

        X_POS = X_REST + u[j] * DISPLAY_SCALE

        x_anchor = X_REST[0] - REST_SPACING
        spring_x1 = [x_anchor] + list(X_POS[:-1])
        for x1, x2 in zip(spring_x1, X_POS):
            draw_spring(ax, x1, x2, 0.0)

        ax.plot(X_POS, np.zeros(N_DOFS), "o", markersize=14, color="k", zorder=3)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"mdof1d/frame_{j // PLOT_EVERY}.jpg")
        plt.close()
