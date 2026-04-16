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

# --------------------------- evolution strategy -------------------------
N, G = 10, 40  # population size, generations
SIGMA0, ELITE = 0.5, 2


def evolve():
    population = np.array(guess) + rng.normal(0, 1.0, size=(N, 2))
    sigma = SIGMA0
    history = [population.copy()]
    for _ in range(G):
        idx = np.argsort(f(population.T))
        parents = population[idx[:ELITE]]
        children = parents[rng.integers(0, ELITE, size=N - ELITE)] + rng.normal(
            0, sigma, size=(N - ELITE, 2)
        )
        population = np.vstack([parents, children])
        sigma *= 0.95
        history.append(population.copy())
    return history


history = evolve()
best = history[-1][np.argmin(f(history[-1].T))]
print(f"best: x={best[0]:.2e}, y={best[1]:.2e}")

# ---------------------------- postprocessing ----------------------------
resolution = 800
x = np.linspace(*xrange, resolution)
y = np.linspace(*yrange, resolution)
xx, yy = np.meshgrid(x, y, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
ax.contourf(xx, yy, z, levels=36, cmap="cividis")
for k, P in enumerate(history):
    ax.plot(
        P[:, 0],
        P[:, 1],
        "o",
        ms=3,
        color=plt.cm.Greys(0.2 + 0.8 * k / len(history)),
        alpha=0.6,
    )
ax.plot(*best, "ro", ms=5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if args.book:
    fig.savefig(RESULTS_DIR / "ea.png")
else:
    plt.show()
plt.close(fig)
