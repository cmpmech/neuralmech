import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ---------------------- settings ----------------------
IMG_SIZE = 400
SAMPLE_RATE = 0.25  # fraction of phase-encode lines to retain
CENTER_FRACTION = 0.08  # always sample this fraction of central k-space lines
N_ITER = 100
TV_WEIGHT = 0.02
SEED = 0

# ------------------------ data ------------------------
rng = np.random.default_rng(SEED)
x_true = resize(shepp_logan_phantom(), (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

kspace_full = np.fft.fft2(x_true)

# variable-density Cartesian mask: fully sampled center, random outer phase-encodes
n_center = max(1, round(CENTER_FRACTION * IMG_SIZE))
center = slice(IMG_SIZE // 2 - n_center // 2, IMG_SIZE // 2 + n_center // 2)
mask_1d = np.zeros(IMG_SIZE, dtype=bool)
mask_1d[center] = True
outer = np.where(~mask_1d)[0]
n_outer = max(0, round(SAMPLE_RATE * IMG_SIZE) - n_center)
mask_1d[rng.choice(outer, n_outer, replace=False)] = True
mask = mask_1d[:, np.newaxis] * np.ones((1, IMG_SIZE), dtype=bool)

kspace_us = kspace_full * mask
x_zero = np.fft.ifft2(kspace_us).real

# -------------------- reconstruction ------------------
# ISTA: gradient of (1/2)||mask * F(x) - kspace_us||^2 + TV proximal
x = x_zero.copy()
for _ in range(N_ITER):
    grad = np.fft.ifft2(mask * (np.fft.fft2(x) - kspace_full)).real
    x = denoise_tv_chambolle(x - grad, weight=TV_WEIGHT)
x_cs = np.clip(x, 0.0, 1.0)

# ------------------- postprocessing -------------------

fig, ax = plt.subplots(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
ax.imshow(x_true.T, origin="lower", cmap="binary", vmin=0, vmax=1)
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / "mri_original.png")
plt.show()

fig, ax = plt.subplots(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
ax.imshow(
    np.log1p(np.abs(np.fft.fftshift(kspace_us))).T, origin="lower", cmap="viridis"
)
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / "mri_measurement.png")
plt.show()

fig, ax = plt.subplots(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
ax.imshow(x_cs.T, origin="lower", cmap="binary", vmin=0, vmax=1)
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / "mri_prediction.png")
plt.show()


# fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# axes[0].imshow(x_true, cmap="binary", vmin=0, vmax=1)
# axes[0].axis("off")

# axes[1].imshow(np.log1p(np.abs(np.fft.fftshift(kspace_us))), cmap="viridis")
# axes[1].axis("off")

# axes[2].imshow(x_zero, cmap="binary", vmin=0, vmax=1)
# axes[2].axis("off")

# axes[3].imshow(x_cs, cmap="binary", vmin=0, vmax=1)
# axes[3].axis("off")

# plt.tight_layout()

# if args.book:
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#     fig.savefig(RESULTS_DIR / "mri.pdf")
#     plt.close()
# else:
#     plt.show()
