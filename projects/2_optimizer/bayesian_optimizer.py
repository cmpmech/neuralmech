import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from optimization_config import ackley as objective
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from postprocessing import save_csv

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

# ------------------------ bayesian optimization -------------------------
N_INIT, N_ITER = 10, 30  # 30  # 30  # initial samples, BO iterations
KAPPA0, KAPPA_MIN = 2.0, 0.1  # initial and final exploration weight for LCB
CANDIDATES = 2000  # candidates for acquisition optimization


def bo():
    X = rng.uniform([xrange[0], yrange[0]], [xrange[1], yrange[1]], size=(N_INIT, 2))
    y = f(X.T)
    for i in range(N_ITER):
        kappa = KAPPA0 * (KAPPA_MIN / KAPPA0) ** (i / (N_ITER - 1))
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=2,
        )
        gp.fit(X, y)
        cand = rng.uniform(
            [xrange[0], yrange[0]], [xrange[1], yrange[1]], size=(CANDIDATES, 2)
        )
        mu, sigma = gp.predict(cand, return_std=True)
        next_x = cand[np.argmin(mu - kappa * sigma)]
        X = np.vstack([X, next_x])
        y = np.append(y, f(next_x[:, None]))
    return X


history = bo()
# cost_history = [np.min(f(pop.T)).item() for pop in history]
costs = f(history.T)
cost_history = np.minimum.accumulate(costs)
best = history[np.argmin(f(history.T))]
print(f"best: x={best[0]:.2e}, y={best[1]:.2e}")

# ---------------------------- postprocessing ----------------------------
resolution = 800
x = np.linspace(*xrange, resolution)
y = np.linspace(*yrange, resolution)
xx, yy = np.meshgrid(x, y, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

# print(history[-1].shape)

fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
ax.contourf(xx, yy, z, levels=36, cmap="cividis")
for k, P in enumerate(history):
    ax.plot(
        P[0],
        P[1],
        "o",
        ms=3,
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
    fig.savefig(RESULTS_DIR / "bo.png")
else:
    plt.show()
plt.close(fig)

# print(cost_history.shape)

# cost history
if args.book:
    save_csv(
        RESULTS_DIR / "bo_history.csv",
        x=np.arange(1, N_INIT + N_ITER + 1),
        y=cost_history,
    )
else:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(cost_history, "k")
    plt.show()
