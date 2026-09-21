# TODO find two nonlinearly separable functions?

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 100  # curves per family
POINTS = 100  # samples per curve
NOISE = 0.02
METHOD = "svd"  # svd, pca or umap
NEIGHBORS = 15  # umap neighborhood size
MIN_DIST = 0.1  # umap spacing of the embedded points
AMPLITUDE = 0.6  # of the oscillating families, so that no family dominates
COLORS = ["k", "r", "b", "g", "m"]

x = np.linspace(0, 1, POINTS)

# every family is drawn with two parameters a and b, both uniform on [0, 1]
FUNCTIONS = {
    "power": lambda a, b: x ** (0.2 + 0.6 * a),
    "linear": lambda a, b: (0.5 + a) * x,
    "kink": lambda a, b: np.minimum((1 + a) * x, 0.3 + 0.1 * x),
    "sine_small": lambda a, b: 0.5 * AMPLITUDE * np.sin(2 * np.pi * (x + a)),
    "sine_large": lambda a, b: AMPLITUDE * np.sin(2 * np.pi * (x + a)),
}


# ------------------------------------ create data ------------------------------------
def draw():
    return rng.random((SAMPLES, 1))


X = np.concatenate([f(draw(), draw()) for f in FUNCTIONS.values()])
X += NOISE * rng.standard_normal(X.shape)
labels = np.repeat(np.arange(len(FUNCTIONS)), SAMPLES)

# ------------------------------ dimensionality reduction -----------------------------
# each curve is one sample, each abscissa one feature
if METHOD == "umap":
    reducer = UMAP(n_neighbors=NEIGHBORS, min_dist=MIN_DIST, random_state=0)
    scores = reducer.fit_transform(X)
else:
    mean = X.mean(axis=0) if METHOD == "pca" else 0  # the svd keeps the mean
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    scores = U[:, :2] * S[:2]
    retained = 100 * np.sum(S[:2] ** 2) / np.sum(S**2)
    print(f"first two components hold {retained:.1f} % of the variance")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, (ax_curves, ax_scores) = plt.subplots(1, 2, figsize=(11, 5))
    for label, color in enumerate(COLORS):
        ax_curves.plot(x, X[labels == label].T, color, alpha=0.3, linewidth=0.5)
        ax_scores.plot(*scores[labels == label].T, color, marker="o", linestyle="")
    ax_scores.set_aspect("equal")  # the two sine families only differ by their radius

    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for label, name in enumerate(FUNCTIONS):
        save_csv(
            CSV_DIR / f"{METHOD}_2D_{name}.csv",
            c1=scores[labels == label, 0],
            c2=scores[labels == label, 1],
        )
