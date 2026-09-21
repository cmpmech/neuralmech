import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
N = 200
N_TREES = [1, 500]
RANDOM_STATE = 0
RESOLUTION = 300

# ------------------------------------ prepare data -----------------------------------
X, y = make_moons(n_samples=N, noise=0.3, random_state=RANDOM_STATE)

# ---------------------------------- decision boundary --------------------------------
x1_grid = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, RESOLUTION)
x2_grid = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, RESOLUTION)
x1g, x2g = np.meshgrid(x1_grid, x2_grid)
X_grid = np.stack([x1g.flatten(), x2g.flatten()], axis=1)

# ------------------------------------- training --------------------------------------
predictions = {}
for n_trees in N_TREES:
    if n_trees == 1:
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    else:
        # random feature subset at each split (here 1 of the 2)
        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=RANDOM_STATE,
            max_samples=100,
        )
        # bagging alone: every split sees both features (with 2D -> similar behavior)
        # model = BaggingClassifier(
        #     DecisionTreeClassifier(random_state=RANDOM_STATE),
        #     n_estimators=n_trees,
        #     max_samples=100,
        # )
    model.fit(X, y)
    predictions[n_trees] = model.predict(X_grid).reshape(RESOLUTION, RESOLUTION)

# ----------------------------------- postprocessing ----------------------------------
for n_trees in N_TREES:
    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=300)
    ax.contourf(x1g, x2g, predictions[n_trees], alpha=0.3, cmap="seismic_r", levels=1)
    id0 = y == 0
    id1 = y == 1
    ax.scatter(X[id0, 0], X[id0, 1], c="r", s=10)
    ax.scatter(X[id1, 0], X[id1, 1], c="b", s=10, marker="s")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if args.book:
        fig.savefig(RGB_PDF_DIR / f"random_forest_{n_trees}.pdf")
        plt.close()
    else:
        plt.show()
