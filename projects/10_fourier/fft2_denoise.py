import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

rng = np.random.default_rng(0)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
NOISE = 20
RADIUS = 150  # keep frequencies within this radius of the spectrum center

# ------------------------------------- load image ------------------------------------
img = Image.open(DATA_DIR / "images" / "oslo.jpg").convert("L")
img_arr = np.array(img, dtype=float)

img_arr += rng.normal(0, NOISE, img_arr.shape)
img_arr = np.clip(img_arr, 0, 255)

# --------------------------------------- helper --------------------------------------
def plot_spectrum(field):
    H, W = field.shape
    x, y = np.arange(W), np.arange(H)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.pcolormesh(X, Y, field, cmap="cividis")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


# --------------------------------------- 2D FFT --------------------------------------
F = np.fft.fft2(img_arr)
F_shifted = np.fft.fftshift(F)

magnitude = np.log1p(np.abs(F_shifted))
phase = np.angle(F_shifted)

# ------------------------------------- filtering -------------------------------------
H, W = img_arr.shape
cy, cx = H // 2, W // 2  # center of the shifted spectrum
Y_grid, X_grid = np.ogrid[:H, :W]
dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)

mask = dist <= RADIUS  # true = low frequency, keep these

F_filtered = F_shifted * mask
magnitude_filtered = np.log1p(np.abs(F_filtered))

img_reconstructed = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
img_reconstructed = np.clip(img_reconstructed, 0, 255)

# ----------------------------------- postprocessing ----------------------------------
fig_freq = plot_spectrum(magnitude)
fig_phase = plot_spectrum(phase)
fig_freq_filtered = plot_spectrum(magnitude_filtered)

if not args.book:
    plt.show()
    show_image(img_arr.astype(np.uint8), grayscale=True)
    show_image(img_reconstructed.astype(np.uint8), grayscale=True)
# -------------------------------- book postprocessing --------------------------------
else:
    fig_freq.savefig(RGB_PDF_DIR / "fft2_freq_denoise.pdf")
    fig_freq_filtered.savefig(RGB_PDF_DIR / "fft2_freq_denoise_filtered.pdf")
    plt.close("all")
    show_image(
        img_arr.astype(np.uint8),
        grayscale=True,
        path=RGB_PDF_DIR / "fft_denoise_og.pdf",
        close=True,
    )
    show_image(
        img_reconstructed.astype(np.uint8),
        grayscale=True,
        path=RGB_PDF_DIR / "fft_denoise_compressed.pdf",
        close=True,
    )
