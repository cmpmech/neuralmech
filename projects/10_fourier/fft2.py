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

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
IMAGES = ["duckling.jpg", "abiskojaure.jpg"]
KEEP_RATIOS = [0.05, 0.2]


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


def compress(F_shifted, keep_ratio):
    abs_F = np.abs(F_shifted)
    threshold = np.percentile(abs_F, (1 - keep_ratio) * 100)
    F_trunc = F_shifted * (abs_F >= threshold)
    img_reconstructed = np.fft.ifft2(np.fft.ifftshift(F_trunc)).real
    return np.clip(img_reconstructed, 0, 255)


for case, image in enumerate(IMAGES):
# ------------------------------------- load image ------------------------------------
    img = Image.open(DATA_DIR / "images" / image).convert("L")
    img_arr = np.array(img, dtype=float)

# --------------------------------------- 2D FFT --------------------------------------
    F = np.fft.fft2(img_arr)
    F_shifted = np.fft.fftshift(F)

    magnitude = np.log1p(np.abs(F_shifted))
    phase = np.angle(F_shifted)

# ----------------------------------- postprocessing ----------------------------------
    fig_freq = plot_spectrum(magnitude)
    fig_phase = plot_spectrum(phase)

    if not args.book:
        plt.show()
        show_image(img_arr.astype(np.uint8), grayscale=True)
        for keep_ratio in KEEP_RATIOS:
            show_image(compress(F_shifted, keep_ratio).astype(np.uint8), grayscale=True)
# -------------------------------- book postprocessing --------------------------------
    else:
        fig_freq.savefig(RGB_PDF_DIR / f"fft2_freq_{case}.pdf")
        fig_phase.savefig(RGB_PDF_DIR / f"fft2_phase_{case}.pdf")
        plt.close("all")
        show_image(
            img_arr.astype(np.uint8),
            grayscale=True,
            path=RGB_PDF_DIR / f"fft_og_{case}.pdf",
            close=True,
        )
        for keep_ratio in KEEP_RATIOS:
            show_image(
                compress(F_shifted, keep_ratio).astype(np.uint8),
                grayscale=True,
                path=RGB_PDF_DIR / f"fft_compressed_{case}_{keep_ratio}.pdf",
                close=True,
            )
