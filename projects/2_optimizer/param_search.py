import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimization_config import ackley as objective

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(42)

# -------------------------------------- settings -------------------------------------
f, xrange, yrange = objective.f, objective.xrange, objective.yrange

SAMPLES = 20**2
MARGIN = 0.1


# ---------------------------------- parameter search ---------------------------------
def structured(n):
    k = int(np.sqrt(n))
    x = np.linspace(xrange[0] + MARGIN, xrange[1] - MARGIN, k)
    y = np.linspace(yrange[0] + MARGIN, yrange[1] - MARGIN, k)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def unstructured(n):
    lo = (xrange[0] + MARGIN, yrange[0] + MARGIN)
    hi = (xrange[1] - MARGIN, yrange[1] - MARGIN)
    return rng.uniform(lo, hi, (n, 2))


# ----------------------------------- postprocessing ----------------------------------
def plot(X, best, name, resolution=800):
    x = np.linspace(*xrange, resolution)
    y = np.linspace(*yrange, resolution)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = f(np.stack([xx, yy], axis=0))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=(resolution // 4))
    ax.contourf(xx, yy, z, levels=48, cmap="cividis")
    ax.plot(X[:, 0], X[:, 1], "ko", ms=4)
    ax.plot(best[0], best[1], "ro", ms=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)  # avoid contourline artifacts
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if args.book:
        fig.savefig(RESULTS_DIR / f"param_search_{name}.png")
    else:
        plt.show()
    plt.close(fig)


for name, sampler in [("structured", structured), ("unstructured", unstructured)]:
    X = sampler(SAMPLES)
    best = X[np.argmin(f(X.T))]
    print(f"best {name}: x={best[0]:.3f}, y={best[1]:.3f}")
    plot(X, best, name)
