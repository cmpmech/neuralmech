from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

# ------------------------------ load image ------------------------------
# img = Image.open(BASE_DIR / "../../data/images/duckling.jpg").convert("L")
img = Image.open(BASE_DIR / "../../data/images/oslo.jpg").convert("L")
# img = Image.open(BASE_DIR / "../../data/images/firebrigade.jpg").convert("L")


# img = Image.open("output.jpg").convert("L")
img_arr = np.array(img, dtype=float)

noise = np.random.normal(0, 20, img_arr.shape)
img_arr += noise
img_arr = np.clip(img_arr, 0, 255)

# -------------------------------- 2D FFT --------------------------------
F = np.fft.fft2(img_arr)
F_shifted = np.fft.fftshift(F)

magnitude = np.log1p(np.abs(F_shifted))
phase = np.angle(F_shifted)

# ---------------- postprocessing of frequencies & phases ----------------
H, W = img_arr.shape
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)

W, H = img_arr.shape[0], img_arr.shape[1]
fig, ax = plt.subplots(figsize=(H / 100, W / 100), dpi=100)
cb = ax.pcolormesh(X, Y, magnitude, cmap="cividis")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / f"fft2_freq_denoise.png")
plt.show()

fig, ax = plt.subplots(figsize=(H / 100, W / 100), dpi=100)
cb = ax.pcolormesh(X, Y, phase, cmap="cividis")
plt.axis("off")
plt.tight_layout(pad=0)
# plt.savefig('../../results/fft2_phase.jpg')
plt.show()

# ------------------------------ filtering -------------------------------
cy, cx = W // 2, H // 2  # center of shifted FFT
# radius = 150  # keep frequencies within this radius
radius = 150  # keep frequencies within this radius

Y_grid, X_grid = np.ogrid[: img_arr.shape[0], : img_arr.shape[1]]
dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)

mask = dist <= radius  # True = low frequency, keep these

F_filtered = F_shifted * mask
magnitude = np.log1p(np.abs(F_filtered))

W, H = img_arr.shape[0], img_arr.shape[1]
fig, ax = plt.subplots(figsize=(H / 100, W / 100), dpi=100)
cb = ax.pcolormesh(X, Y, magnitude, cmap="cividis")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / f"fft2_freq_denoise_filtered.png")
plt.show()

img_reconstructed = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
img_reconstructed = np.clip(img_reconstructed, 0, 255)

show_image(
    img_arr.astype(np.uint8), grayscale=True, path=RESULTS_DIR / f"fft_denoise_og.jpg"
)
show_image(
    img_reconstructed.astype(np.uint8),
    grayscale=True,
    path=RESULTS_DIR / f"fft_denoise_compressed.jpg",
)
