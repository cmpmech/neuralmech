import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
CLASSES = ["circle", "ellipse", "square", "triangle", "star", "cross"]
NEIGHBORS = 15  # umap neighborhood size
MIN_DIST = 0.1  # umap spacing of the embedded points

# ------------------------------------- load data -------------------------------------
images = [np.load(DATA_DIR / f"shapes_{name}_128.npy") for name in CLASSES]
X = np.concatenate(images).reshape(-1, 128 * 128)  # each image is one sample
labels = np.repeat(np.arange(len(CLASSES)), len(images[0]))

# ------------------------------ dimensionality reduction -----------------------------
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
pca = U[:, :2] * S[:2]
retained = 100 * np.sum(S[:2] ** 2) / np.sum(S**2)
print(f"first two components hold {retained:.1f} % of the variance")

umap = UMAP(n_neighbors=NEIGHBORS, min_dist=MIN_DIST, random_state=0).fit_transform(X)

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for label in range(len(CLASSES)):
        axes[0].plot(*pca[labels == label].T, "o", markersize=2)
        axes[1].plot(*umap[labels == label].T, "o", markersize=2)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for label, name in enumerate(CLASSES):
        for method, Z in [("pca", pca), ("umap", umap)]:
            Z = Z[labels == label]
            save_csv(CSV_DIR / f"shapes_{method}_{name}.csv", z1=Z[:, 0], z2=Z[:, 1])
