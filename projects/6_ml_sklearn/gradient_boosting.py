import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 64
TREES = [1, 10, 100, 1000]
DEPTH = 2
LR = 0.1


# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

X = np.stack([x1.flatten(), x2.flatten()], axis=1)
y = np.sin(1 * np.pi * x1) * np.sin(2 * np.pi * x1 * x2)
Y = y.flatten()


# ------------------------------------- training --------------------------------------
predictions = {}
for trees in TREES:
    model = GradientBoostingRegressor(
        n_estimators=trees, max_depth=DEPTH, learning_rate=LR
    )
    model.fit(X, Y)
    predictions[trees] = model.predict(X).reshape(RESOLUTION, RESOLUTION)
    rmse = np.sqrt(np.mean((predictions[trees] - y) ** 2))
    print(f"{trees} trees: rmse {rmse:.2e}")


# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(TREES))
    for ax, trees in zip(axes, TREES):
        ax.imshow(predictions[trees], vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for trees in TREES:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=200)
        ax.imshow(predictions[trees], vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"gradient_boosting_{trees}.pdf")
        plt.close(fig)
