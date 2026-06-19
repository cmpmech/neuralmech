import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()
# -------------------------------------- settings -------------------------------------
RESOLUTION = 64  # 64
DEPTH = 10
N_TREES = [1, 2, 4, 8, 32]
# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")
X = np.stack([x1.flatten(), x2.flatten()], axis=1)
y = np.sin(1 * np.pi * x1) * np.sin(2 * np.pi * x1 * x2)
Y = y.flatten()
# ------------------------------------- training --------------------------------------
predictions = {}
for n_trees in N_TREES:
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=DEPTH, random_state=0)
    model.fit(X, Y)
    predictions[n_trees] = model.predict(X).reshape(RESOLUTION, RESOLUTION)
# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(N_TREES))
    for ax, n_trees in zip(axes, N_TREES):
        ax.imshow(predictions[n_trees], vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n_trees in N_TREES:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=200)
        ax.imshow(predictions[n_trees], vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RESULTS_DIR / f"random_forest_{n_trees}.png")
        plt.close(fig)
