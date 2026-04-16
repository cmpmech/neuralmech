import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from optimization_config import ackley as objective
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(3)
f, xrange, yrange = objective.f, objective.xrange, objective.yrange

# ----------------------- bayesian optimization --------------------------
N_INIT, N_ITER, N_CAND = 5, 35, 2000


def expected_improvement(candidates, gp, y_best):
    mu, sigma = gp.predict(candidates, return_std=True)
    z = (y_best - mu) / (sigma + 1e-9)
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def bo():
    lo = np.array([xrange[0], yrange[0]])
    hi = np.array([xrange[1], yrange[1]])
    X = rng.uniform(lo, hi, (N_INIT, 2))
    y = f(X.T)
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)
    for _ in range(N_ITER):
        gp.fit(X, y)
        candidates = rng.uniform(lo, hi, (N_CAND, 2))
        ei = expected_improvement(candidates, gp, y.min())
        x_next = candidates[np.argmax(ei)]
        X = np.vstack([X, x_next])
        y = np.append(y, f(x_next))
    return X, y


X, y = bo()
best = X[np.argmin(y)]
print(f"best: x={best[0]:.2e}, y={best[1]:.2e}")

# ---------------------------- postprocessing ----------------------------
resolution = 800
x = np.linspace(*xrange, resolution)
yg = np.linspace(*yrange, resolution)
xx, yy = np.meshgrid(x, yg, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
ax.contourf(xx, yy, z, levels=36, cmap="cividis")
for k, pt in enumerate(X):
    ax.plot(
        pt[0],
        pt[1],
        "o",
        ms=3,
        color=plt.cm.Greys(0.2 + 0.8 * k / len(X)),
        alpha=0.7,
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
