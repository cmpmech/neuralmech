import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 400
SAMPLE_RATE = 0.25  # fraction of phase-encode lines to retain
CENTER_FRACTION = 0.08  # always sample this fraction of central k-space lines
ITERS = 100
TV_WEIGHT = 0.02
SEED = 0

# ------------------------------------ create data ------------------------------------
rng = np.random.default_rng(SEED)
x_true = resize(shepp_logan_phantom(), (RESOLUTION, RESOLUTION), anti_aliasing=True)

kspace_full = np.fft.fft2(x_true)

# sampling
n_center = max(1, round(CENTER_FRACTION * RESOLUTION))
center = slice(RESOLUTION // 2 - n_center // 2, RESOLUTION // 2 + n_center // 2)
mask_1d = np.zeros(RESOLUTION, dtype=bool)
mask_1d[center] = True
outer = np.where(~mask_1d)[0]
n_outer = max(0, round(SAMPLE_RATE * RESOLUTION) - n_center)
mask_1d[rng.choice(outer, n_outer, replace=False)] = True
mask = mask_1d[:, np.newaxis] * np.ones((1, RESOLUTION), dtype=bool)

kspace_us = kspace_full * mask
x_zero = np.fft.ifft2(kspace_us).real

# ----------------------------------- reconstruction ----------------------------------
x = x_zero.copy()
for _ in range(ITERS):
    grad = np.fft.ifft2(mask * (np.fft.fft2(x) - kspace_full)).real
    x = denoise_tv_chambolle(x - grad, weight=TV_WEIGHT)
x_cs = np.clip(x, 0.0, 1.0)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
ax.imshow(x_true.T, origin="lower", cmap="binary", vmin=0, vmax=1)
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "mri_original.png")
plt.show()

fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
ax.imshow(
    np.log1p(np.abs(np.fft.fftshift(kspace_us))).T, origin="lower", cmap="viridis"
)
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "mri_measurement.png")
plt.show()

fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
ax.imshow(x_cs.T, origin="lower", cmap="binary", vmin=0, vmax=1)
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "mri_prediction.png")
plt.show()
