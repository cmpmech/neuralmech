import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from optimization_config import ackley as objective
from postprocessing import save_csv

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

N_INIT, N_ITER = 10, 30  # initial samples, bayesian optimization iterations
KAPPA0, KAPPA_MIN = 2.0, 0.1  # initial and final exploration weight
CANDIDATES = 2000  # candidates for acquisition optimization


# ------------------------------- bayesian optimization -------------------------------
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
        next_x = cand[np.argmin(mu - kappa * sigma)]  # choose smallest
        X = np.vstack([X, next_x])
        y = np.append(y, f(next_x[:, None]))
    return X


history = bo()
costs = f(history.T)
cost_history = np.minimum.accumulate(costs)
best = history[np.argmin(f(history.T))]
print(f"best: x={best[0]:.2e}, y={best[1]:.2e}")

# ----------------------------------- postprocessing ----------------------------------
resolution = 800
x = np.linspace(*xrange, resolution)
y = np.linspace(*yrange, resolution)
xx, yy = np.meshgrid(x, y, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution // 4)
ax.contourf(xx, yy, z, levels=36, cmap="cividis")
for k, P in enumerate(history):
    ax.plot(P[0], P[1], "o", ms=4, color=plt.cm.Greys(0.2 + 0.8 * k / len(history)))
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
    fig.savefig(RGB_PDF_DIR / "bo.pdf")
    save_csv(
        CSV_DIR / "bo_history.csv",
        x=np.arange(1, N_INIT + N_ITER + 1),
        y=cost_history,
    )
plt.close("all")
