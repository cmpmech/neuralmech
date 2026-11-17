import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../external_data/yale_faces").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RANKS = [25, 100, 400, 1600]
EIGENFACES = 4
EYES_YALE = (32 + 45j, 132 + 44j)  # eyes of mean face (column + i row)
PHOTOS = {"shaved": (348 + 380j, 828 + 380j), "smiling": (345 + 380j, 832 + 380j)}  # pupils

# ------------------------------------- load data -------------------------------------
data = scipy.io.loadmat(DATA_DIR / "allFaces.mat")  # extended Yale face database B
n, m = int(data["n"][0, 0]), int(data["m"][0, 0])
X = data["faces"].T / 255  # one column-major 192 x 168 image per row


def align(path, eyes):
    a = (eyes[1] - eyes[0]) / (EYES_YALE[1] - EYES_YALE[0])  # similarity map from face to photo
    b = eyes[0] - a * EYES_YALE[0]
    photo = Image.open(path).convert("L")
    face = photo.transform((m, n), Image.AFFINE, (a.real, -a.imag, b.real, a.imag, a.real, b.imag), Image.BICUBIC)
    x = np.asarray(face).T.flatten() / 255
    return (x - x.mean()) / x.std() * X.std(axis=1).mean() + X.mean()  # brightness and contrast of dataset


x_test = {name: align(DATA_DIR / f"leon_{name}.png", eyes) for name, eyes in PHOTOS.items()}

# ----------------------------- principal component analysis --------------------------
mean = X.mean(axis=0)
U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)


def reconstruct(x, rank):
    z = Vt[:rank] @ (x - mean)
    return mean + Vt[:rank].T @ z


for name, x in x_test.items():
    for rank in RANKS:
        error = np.linalg.norm(reconstruct(x, rank) - x) / np.linalg.norm(x)
        print(f"{name} R = {rank}: error {error:.3f}")

# ----------------------------------- postprocessing ----------------------------------
images = [("mean", mean, False)]
images += [(f"mode_{i + 1}", Vt[i], True) for i in range(EIGENFACES)]  # signed
for name, x in x_test.items():
    images += [(f"{name}_rank_{rank}", reconstruct(x, rank), False) for rank in RANKS]
    images += [(f"{name}_original", x, False)]

if not args.book:
    fig, axes = plt.subplots(3, 5, figsize=(10, 8))
    for ax, (name, image, signed) in zip(axes.flat, images):
        ax.imshow(image.reshape(m, n).T, cmap="gray")
        ax.set_title(name)
    for ax in axes.flat:
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, image, signed in images:
        limit = np.abs(image).max()
        limits = (-limit, limit) if signed else (0, 1)
        fig, ax = plt.subplots(figsize=(m / 100, n / 100), dpi=100)
        ax.imshow(image.reshape(m, n).T, cmap="gray", vmin=limits[0], vmax=limits[1])
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"eigenfaces_{name}.pdf")
        plt.close(fig)
