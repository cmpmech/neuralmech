import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ---------------------- settings ----------------------
SAMPLE_RATE = 0.25  # fraction of k-space coefficients to keep
N_ITER = 300
TV_WEIGHT = 0.5
IMG_SIZE = 256
SEED = 0

# ------------------------ data ------------------------
rng = np.random.default_rng(SEED)
img = Image.open(BASE_DIR / "../../data/images/rewe.jpg").convert("L")
img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
x_true = np.array(img, dtype=float)

# random binary mask selects SAMPLE_RATE fraction of Fourier coefficients
mask = rng.random((IMG_SIZE, IMG_SIZE)) < SAMPLE_RATE
y = mask * np.fft.fft2(x_true)

# zero-filled inverse FFT: naive baseline with aliasing artifacts
x_zero = np.fft.ifft2(y).real

# -------------------- reconstruction ------------------
# ISTA: gradient step on data fidelity + TV proximal operator
# data fidelity: (1/2) ||mask * F(x) - y||^2, Lipschitz constant L=1
x = x_zero.copy()
for _ in range(N_ITER):
    grad = np.fft.ifft2(mask * (np.fft.fft2(x) - y)).real
    x = denoise_tv_chambolle(x - grad, weight=TV_WEIGHT)

x_cs = np.clip(x, 0, 255)

# ------------------- postprocessing -------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(x_true, cmap="gray", vmin=0, vmax=255)
axes[0].axis("off")

axes[1].imshow(x_zero, cmap="gray", vmin=0, vmax=255)
axes[1].axis("off")

axes[2].imshow(x_cs, cmap="gray", vmin=0, vmax=255)
axes[2].axis("off")

plt.tight_layout()

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "compressed_sensing.pdf")
    plt.close()
else:
    plt.show()
