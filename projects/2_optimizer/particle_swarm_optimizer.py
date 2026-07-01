import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimization_config import ackley as objective
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(3)

# -------------------------------------- settings -------------------------------------
f, xrange, yrange, guess = (
    objective.f,
    objective.xrange,
    objective.yrange,
    objective.guess,
)

N, G = 10, 40  # particles, iterations
W, C1, C2 = 0.5, 0.1, 0.3  # inertia, cognitive, social


# ------------------------------- particle swarm optimizer ----------------------------
def pso():
    lo = np.array([xrange[0], yrange[0]])
    hi = np.array([xrange[1], yrange[1]])
    pos = rng.uniform(lo, hi, (N, 2))
    vel = rng.uniform(-(hi - lo), hi - lo, (N, 2))
    pbest = pos.copy()
    gbest = pbest[np.argmin(f(pbest.T))]
    history = [pos.copy()]
    for _ in range(G):
        r1, r2 = rng.random((N, 2)), rng.random((N, 2))
        vel = W * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)
        pos = pos + vel
        improved = f(pos.T) < f(pbest.T)
        pbest[improved] = pos[improved]
        gbest = pbest[np.argmin(f(pbest.T))]
        history.append(pos.copy())
    return history, gbest


history, best = pso()
cost_history = [np.min(f(P.T)).item() for P in history]
print(f"best: x={best[0]:.2e}, y={best[1]:.2e}")

# ----------------------------------- postprocessing ----------------------------------
resolution = 800
x = np.linspace(*xrange, resolution)
y = np.linspace(*yrange, resolution)
xx, yy = np.meshgrid(x, y, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
ax.contourf(xx, yy, z, levels=36, cmap="cividis")
for i in range(N):
    for k in range(len(history) - 1):
        ax.plot(
            [history[k][i, 0], history[k + 1][i, 0]],
            [history[k][i, 1], history[k + 1][i, 1]],
            "-o",
            ms=4,
            lw=2,
            color=plt.cm.Greys(0.2 + 0.8 * k / len(history)),
        )
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(cost_history, "k")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig.savefig(RGB_PDF_DIR / "pso.pdf")
    save_csv(CSV_DIR / "pso_history.csv", x=np.arange(0, G + 1), y=cost_history)
plt.close("all")
