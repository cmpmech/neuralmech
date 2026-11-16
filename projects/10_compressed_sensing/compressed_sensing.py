import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dct, dctn, idct, idctn

from postprocessing import save_csv, show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

rng = np.random.default_rng(0)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# signal
N = 4096  # samples of the full signal on [0, 1]
FREQUENCIES = [97, 777]
SAMPLES = 128  # random measurements
SAMPLE_SWEEP = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
WINDOW = 0.03  # shown part of the signal

# image
RESOLUTION = 256
KEPT = 0.2  # fraction of measured pixels

# sparse regression
REGULARIZATION = 0.01
ITERS = 20000  # plain ISTA converges slowly for small REGULARIZATION

t = (np.arange(N) + 0.5) / N  # midpoints, where the cosines are discrete cosine modes
signal = sum(np.cos(2 * np.pi * f * t) for f in FREQUENCIES)


# --------------------------------------- helper --------------------------------------
def ista(y, forward, adjoint, shape):
    s = np.zeros(shape)
    for _ in range(ITERS):
        s = s + adjoint(y - forward(s))  # unit step, the operators have norm 1
        s = np.sign(s) * np.maximum(np.abs(s) - REGULARIZATION, 0)
    return s


def sense_signal(samples):
    ids = np.sort(rng.choice(N, samples, replace=False))
    Psi = idct(np.eye(N), axis=0, norm="ortho")
    Theta = Psi[ids]
    y = signal[ids]
    s_pinv = np.linalg.pinv(Theta) @ y
    s_l1 = ista(y, lambda s: Theta @ s, lambda r: Theta.T @ r, N)
    return ids, Psi @ s_pinv, Psi @ s_l1, s_pinv, s_l1


def relative_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


# ------------------------------------ 1D signal --------------------------------------
ids, x_pinv, x_l1, s_pinv, s_l1 = sense_signal(SAMPLES)
print(f"{SAMPLES} samples: pseudoinverse error {relative_error(x_pinv, signal):.2e}")
print(f"{SAMPLES} samples: l1 error {relative_error(x_l1, signal):.2e}")

errors = []
for samples in SAMPLE_SWEEP:
    _, sweep_pinv, sweep_l1, _, _ = sense_signal(samples)
    errors.append(
        [relative_error(sweep_pinv, signal), relative_error(sweep_l1, signal)]
    )
    print(f"{samples} samples: errors {errors[-1][0]:.2e}, {errors[-1][1]:.2e}")
errors = np.array(errors)

# ------------------------------------- 2D image --------------------------------------
img = Image.open(DATA_DIR / "images/swan.jpg").convert("L")
side = min(img.size)
img = img.crop(((img.width - side) // 2, 0, (img.width + side) // 2, side))
img = np.asarray(img.resize((RESOLUTION, RESOLUTION)), dtype=float) / 255

measured = rng.random(img.shape) < KEPT
y = img[measured]


def forward(s):
    return idctn(s, norm="ortho")[measured]


def adjoint(r):
    image = np.zeros(img.shape)
    image[measured] = r
    return dctn(image, norm="ortho")


img_pinv = idctn(adjoint(y), norm="ortho")  # zero filled, the minimum-norm solution
img_l1 = np.clip(idctn(ista(y, forward, adjoint, img.shape), norm="ortho"), 0, 1)
print(f"image: zero-filled error {relative_error(img_pinv, img):.3f}")
print(f"image: l1 error {relative_error(img_l1, img):.3f}")

# ----------------------------------- postprocessing ----------------------------------
img_measured = np.where(measured, img, 0.5)
images = {"original": img, "measured": img_measured, "pinv": img_pinv, "l1": img_l1}
shown = t < WINDOW
spectrum = dct(signal, norm="ortho")

if not args.book:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t[shown], signal[shown], "k--")
    axes[0].plot(t[shown], x_pinv[shown], "b")
    axes[0].plot(t[shown], x_l1[shown], "r")
    axes[0].plot(t[ids][t[ids] < WINDOW], signal[ids][t[ids] < WINDOW], "ko")
    axes[1].plot(np.abs(spectrum), "k--")
    axes[1].plot(np.abs(s_pinv), "b")
    axes[1].plot(np.abs(s_l1), "r")
    axes[2].semilogy(SAMPLE_SWEEP, errors, "o-")
    plt.show()
    for image in images.values():
        show_image(image, grayscale=True)
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "cs_signal.csv",
        t=t[shown],
        exact=signal[shown],
        pinv=x_pinv[shown],
        l1=x_l1[shown],
    )
    sampled = ids[t[ids] < WINDOW]
    save_csv(CSV_DIR / "cs_samples.csv", t=t[sampled], y=signal[sampled])
    save_csv(
        CSV_DIR / "cs_spectrum.csv",
        frequency=np.arange(N) / 2,
        exact=np.abs(spectrum),
        pinv=np.abs(s_pinv),
        l1=np.abs(s_l1),
    )
    save_csv(
        CSV_DIR / "cs_error.csv",
        samples=SAMPLE_SWEEP,
        pinv=errors[:, 0],
        l1=errors[:, 1],
    )
    for name, image in images.items():
        show_image(
            image, grayscale=True, path=RGB_PDF_DIR / f"cs_{name}.pdf", close=True
        )
