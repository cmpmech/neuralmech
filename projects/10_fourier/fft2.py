from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"


CASE = 1

# ------------------------------ load image ------------------------------
if CASE == 0:
    img = Image.open(BASE_DIR / "../../data/images/duckling.jpg").convert("L")
elif CASE == 1:
    img = Image.open(BASE_DIR / "../../data/images/abiskojaure.jpg").convert("L")

# img = Image.open("output.jpg").convert("L")
img_arr = np.array(img, dtype=float)

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
plt.savefig(RESULTS_DIR / f"fft2_freq_{CASE}.png")
plt.show()

fig, ax = plt.subplots(figsize=(H / 100, W / 100), dpi=100)
cb = ax.pcolormesh(X, Y, phase, cmap="cividis")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / f"fft2_phase_{CASE}.png")
plt.show()

# ------------------------------ truncation ------------------------------
keep_ratio = 0.05
# keep_ratio = 0.1
# keep_ratio = 0.2
abs_F = np.abs(F_shifted)
threshold = np.percentile(abs_F, (1 - keep_ratio) * 100)
mask = abs_F >= threshold

F_trunc = F_shifted * mask

img_reconstructed = np.fft.ifft2(np.fft.ifftshift(F_trunc)).real
img_reconstructed = np.clip(img_reconstructed, 0, 255)

show_image(
    img_arr.astype(np.uint8),
    grayscale=True,
    path=RESULTS_DIR / f"fft_og_{CASE}.jpg",
)
show_image(
    img_reconstructed.astype(np.uint8),
    grayscale=True,
    path=RESULTS_DIR / f"fft_compressed_{CASE}_{keep_ratio}.jpg",
)
