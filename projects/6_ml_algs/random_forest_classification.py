import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()
# -------------------------------------- settings -------------------------------------
N = 500
N_TREES = [1, 500]  # 4, 16, 64]
RANDOM_STATE = 0
RESOLUTION = 400
DEPTH = 50
# ------------------------------------ prepare data -----------------------------------
# X, y = make_moons(n_samples=N, noise=0.25, random_state=RANDOM_STATE)
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
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=DEPTH)
    else:
        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=RANDOM_STATE,
            max_depth=DEPTH,
            max_samples=100,
        )
        # model = BaggingClassifier(
        #     DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=DEPTH),
        #     n_estimators=n_trees,
        #     max_samples=100,
        # )
    model.fit(X, y)
    predictions[n_trees] = model.predict(X_grid).reshape(RESOLUTION, RESOLUTION)
# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(N_TREES))
    for ax, n_trees in zip(axes, N_TREES):
        ax.contourf(x1g, x2g, predictions[n_trees], alpha=0.4, cmap="Spectral")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Spectral", s=20, linewidths=0)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n_trees in N_TREES:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=200)
        ax.contourf(x1g, x2g, predictions[n_trees], alpha=0.4, cmap="Spectral")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Spectral", s=5, linewidths=0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RESULTS_DIR / f"random_forest_{n_trees}.png")
        plt.close(fig)

# import argparse
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

# BASE_DIR = Path(__file__).parent
# RESULTS_DIR = (BASE_DIR / "../../results").resolve()
# parser = argparse.ArgumentParser()
# parser.add_argument("--book", action="store_true")
# args = parser.parse_args()
# # -------------------------------------- settings -------------------------------------
# N = 1000  # 500
# N_TREES = [1, 4, 16, 64, 200]
# RANDOM_STATE = 0
# RESOLUTION = 200
# # ------------------------------------ prepare data -----------------------------------
# rng = np.random.default_rng(RANDOM_STATE)
# x1 = rng.uniform(-1, 1, N)
# x2 = rng.uniform(-1, 1, N)
# noise = rng.normal(0, 0.2, N)
# y = (x2 > np.sin(np.pi * x1) + noise).astype(int)
# X = np.stack([x1, x2], axis=1)
# # ---------------------------------- decision boundary --------------------------------
# x1_grid = np.linspace(-1, 1, RESOLUTION)
# x2_grid = np.linspace(-1, 1, RESOLUTION)
# x1g, x2g = np.meshgrid(x1_grid, x2_grid)
# X_grid = np.stack([x1g.flatten(), x2g.flatten()], axis=1)
# # ------------------------------------- training --------------------------------------
# predictions = {}
# for n_trees in N_TREES:
#     if n_trees == 1:
#         model = DecisionTreeClassifier(random_state=RANDOM_STATE)
#     else:
#         model = RandomForestClassifier(n_estimators=n_trees, random_state=RANDOM_STATE)
#     model.fit(X, y)
#     predictions[n_trees] = model.predict(X_grid).reshape(RESOLUTION, RESOLUTION)
# # ----------------------------------- postprocessing ----------------------------------
# if not args.book:
#     fig, axes = plt.subplots(1, len(N_TREES))
#     for ax, n_trees in zip(axes, N_TREES):
#         ax.contourf(x1g, x2g, predictions[n_trees], alpha=0.4, cmap="Spectral")
#         ax.scatter(x1, x2, c=y, cmap="Spectral", s=5, linewidths=0)
#         ax.axis("off")
#     fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.show()
# # -------------------------------- book postprocessing --------------------------------
# else:
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#     for n_trees in N_TREES:
#         fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=200)
#         ax.contourf(x1g, x2g, predictions[n_trees], alpha=0.4, cmap="Spectral")
#         ax.scatter(x1, x2, c=y, cmap="Spectral", s=5, linewidths=0)
#         ax.axis("off")
#         ax.set_xlim(-1, 1)
#         ax.set_ylim(-1, 1)
#         fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#         fig.savefig(RESULTS_DIR / f"random_forest_{n_trees}.png")
#         plt.close(fig)
