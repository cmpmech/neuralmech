import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RANKS = [5, 20, 50, 150]
EIGENFACES = 4
TEST_PERSON = 0  # all images of this person are held out from the basis

# ------------------------------------- load data -------------------------------------
faces = fetch_olivetti_faces(data_home=DATA_DIR / "olivetti")  # 400 images, 40 people
X = faces.data  # one 64 x 64 image per row
test = faces.target == TEST_PERSON
X_train = X[~test]
x_test = X[test][0]

# ----------------------------- principal component analysis --------------------------
mean = X_train.mean(axis=0)
U, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
z = Vt @ (x_test - mean)  # coordinates of the unseen face in the basis


def reconstruct(rank):
    return mean + Vt[:rank].T @ z[:rank]


variance = np.cumsum(S**2) / np.sum(S**2)
for rank in RANKS:
    error = np.linalg.norm(reconstruct(rank) - x_test) / np.linalg.norm(x_test)
    print(f"R = {rank}: variance kept {variance[rank - 1]:.3f}, error {error:.3f}")

# ----------------------------------- postprocessing ----------------------------------
images = [("mean", mean, False), ("original", x_test, False)]
images += [(f"mode_{i + 1}", Vt[i], True) for i in range(EIGENFACES)]  # signed
images += [(f"rank_{rank}", reconstruct(rank), False) for rank in RANKS]

if not args.book:
    fig, axes = plt.subplots(2, len(images) // 2 + 1, figsize=(12, 5))
    for ax, (name, image, signed) in zip(axes.flat, images):
        ax.imshow(image.reshape(64, 64), cmap="gray")
        ax.set_title(name)
    for ax in axes.flat:
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, image, signed in images:
        limit = np.abs(image).max()
        limits = (-limit, limit) if signed else (0, 1)
        fig, ax = plt.subplots(figsize=(0.64, 0.64), dpi=100)
        ax.imshow(image.reshape(64, 64), cmap="gray", vmin=limits[0], vmax=limits[1])
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"eigenfaces_{name}.pdf")
        plt.close(fig)
    ranks = np.arange(1, len(S) + 1)
    save_csv(CSV_DIR / "eigenfaces_variance.csv", R=ranks, value=variance)
