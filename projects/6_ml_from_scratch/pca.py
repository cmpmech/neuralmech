import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RANKS = [1, 5, 20, 100]

# ------------------------------------- load image ------------------------------------
img = Image.open(DATA_DIR / "images/flying_goose.jpg").convert("L")
img_arr = np.asarray(img, dtype=float) / 255

# ---------------------------- principal component analysis ---------------------------
mean = img_arr.mean(axis=0)  # each image row is treated as one sample
U, S, Vt = np.linalg.svd(img_arr - mean, full_matrices=False)

H, W = img_arr.shape
variance = S**2 / (H - 1)
ratios = [H * W / (rank * (H + W) + W) for rank in RANKS]


def reconstruct(rank):
    approx = mean + U[:, :rank] @ (S[:rank, None] * Vt[:rank, :])
    return np.clip(approx, 0, 1)


# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(RANKS), figsize=(4 * len(RANKS), 4 * H / W))
    for ax, rank, ratio in zip(axes, RANKS, ratios):
        ax.imshow(reconstruct(rank), cmap="gray")
        ax.set_title(f"rank {rank}, {ratio:.0f}x")
        ax.axis("off")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(variance) + 1), variance, "k")

    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "pca_explained_variance.csv",
        idx=np.arange(1, len(variance) + 1),
        value=variance,
    )
