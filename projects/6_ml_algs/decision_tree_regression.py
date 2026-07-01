import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 64
DEPTHS = [1, 2, 4, 8, 16]


# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

X = np.stack([x1.flatten(), x2.flatten()], axis=1)
y = np.sin(1 * np.pi * x1) * np.sin(2 * np.pi * x1 * x2)
Y = y.flatten()


# ------------------------------------- training --------------------------------------
predictions = {}
for depth in DEPTHS:
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(X, Y)
    predictions[depth] = model.predict(X).reshape(RESOLUTION, RESOLUTION)


# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(DEPTHS))
    for ax, depth in zip(axes, DEPTHS):
        ax.imshow(predictions[depth], vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:

    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=200)
    ax.imshow(y, vmin=-1, vmax=1, cmap="Spectral")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RGB_PDF_DIR / "decision_tree_groundtruth.pdf")
    plt.close(fig)

    for depth in DEPTHS:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=200)
        ax.imshow(predictions[depth], vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"decision_tree_{depth}.pdf")
        plt.close(fig)
