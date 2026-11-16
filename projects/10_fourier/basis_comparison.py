import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from postprocessing import save_csv, show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
STORAGE = np.geomspace(1e-3, 0.5, 25)  # stored reals as fraction of the pixel count
SHOWN = 0.02  # storage of the exported reconstructions
BLOCK = 8  # block size of the discrete cosine transform, as in JPEG

# ------------------------------------- load image ------------------------------------
img = Image.open(DATA_DIR / "images/window.jpg").convert("L")
img_arr = np.asarray(img, dtype=float) / 255
H = img_arr.shape[0] // BLOCK * BLOCK
W = img_arr.shape[1] // BLOCK * BLOCK
img_arr = img_arr[:H, :W]


# --------------------------------------- helper --------------------------------------
def keep_largest(coefficients, count):
    threshold = np.sort(np.abs(coefficients), axis=None)[-count]
    return np.where(np.abs(coefficients) >= threshold, coefficients, 0)


def svd(storage):
    rank = max(1, round(storage * H * W / (H + W + 1)))
    return U[:, :rank] * S[:rank] @ Vt[:rank]


def fft(storage):
    # conjugate pairs share their magnitude, so one kept coefficient costs one real
    return np.fft.ifft2(keep_largest(F, round(storage * H * W))).real


def dct(storage):
    kept = keep_largest(D, round(storage * H * W))
    return idctn(kept, axes=(1, 3), norm="ortho").reshape(H, W)


def relative_error(approx):
    return np.linalg.norm(approx - img_arr) / np.linalg.norm(img_arr)


# ------------------------------------- transforms ------------------------------------
U, S, Vt = np.linalg.svd(img_arr, full_matrices=False)
F = np.fft.fft2(img_arr)
blocks = img_arr.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK)
D = dctn(blocks, axes=(1, 3), norm="ortho")

methods = {"svd": svd, "fft": fft, "dct": dct}
errors = {name: [relative_error(f(s)) for s in STORAGE] for name, f in methods.items()}
for name, f in methods.items():
    print(f"{name}: error {relative_error(f(0.01)):.3f} at 1 %, ", end="")
    print(f"{relative_error(f(0.1)):.3f} at 10 % storage")

# ----------------------------------- postprocessing ----------------------------------
reconstructions = {name: np.clip(f(SHOWN), 0, 1) for name, f in methods.items()}

if not args.book:
    fig, ax = plt.subplots()
    for name, color in zip(errors, ["k", "r", "b"]):
        ax.loglog(STORAGE, errors[name], color)
    plt.show()
    for img in reconstructions.values():
        show_image(img, grayscale=True)
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "basis_comparison.csv", storage=STORAGE, **errors)
    for name, img in reconstructions.items():
        show_image(
            img, grayscale=True, path=RGB_PDF_DIR / f"basis_{name}.pdf", close=True
        )
