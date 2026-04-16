import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from optimization_config import ackley as objective

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(3)
f, xrange, yrange, guess = (
    objective.f,
    objective.xrange,
    objective.yrange,
    objective.guess,
)

# ----------------------- simulated annealing ----------------------------
STEPS, T0, ALPHA, SIGMA = 400, 2.0, 0.98, 0.3


def anneal():
    x = np.array(guess, dtype=float)
    fx = f(x)
    T = T0
    history = [x.copy()]
    for _ in range(STEPS):
        candidate = x + rng.normal(0, SIGMA, size=2)
        fc = f(candidate)
        if fc < fx or rng.random() < np.exp((fx - fc) / T):
            x, fx = candidate, fc
        T *= ALPHA
        history.append(x.copy())
    return np.array(history)


history = anneal()
best = history[np.argmin(f(history.T))]
print(f"best: x={best[0]:.2e}, y={best[1]:.2e}")

# ---------------------------- postprocessing ----------------------------
resolution = 800
x = np.linspace(*xrange, resolution)
y = np.linspace(*yrange, resolution)
xx, yy = np.meshgrid(x, y, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
ax.contourf(xx, yy, z, levels=36, cmap="cividis")
for k in range(len(history) - 1):
    ax.plot(
        history[k : k + 2, 0],
        history[k : k + 2, 1],
        "-o",
        ms=2,
        lw=1,
        color=plt.cm.Greys(0.2 + 0.8 * k / len(history)),
        alpha=0.6,
    )
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if args.book:
    fig.savefig(RESULTS_DIR / "sa.png")
else:
    plt.show()
plt.close(fig)
