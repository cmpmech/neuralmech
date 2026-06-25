import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# RESOLUTION = 3
# RESOLUTION = 5
RESOLUTION = 7
ANGLE = 45

# ------------------------------------ create data ------------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

y = np.exp(-(x1**2 + x2**2))
y_rot = scipy.ndimage.rotate(y, ANGLE, reshape=False)

# ----------------------------------- postprocessing ----------------------------------
mae = np.mean(np.abs(y - y_rot))
rmse = np.sqrt(np.mean((y - y_rot) ** 2))
print(f"mae: {mae:.2e}, rmse: {rmse:.2e}")

# rotation error vs grid resolution, restricted to the inscribed disk
resolutions = list(range(3, 33))
maes, rmses = [], []
for res in resolutions:
    x1 = np.linspace(-1, 1, res)
    x2 = np.linspace(-1, 1, res)
    x1, x2 = np.meshgrid(x1, x2, indexing="ij")
    mask = (x1**2 + x2**2) <= 1.0

    y = np.exp(-(x1**2 + x2**2))
    y_rot = scipy.ndimage.rotate(y, ANGLE, reshape=False)

    maes.append(np.mean(np.abs(y[mask] - y_rot[mask])))
    rmses.append(np.sqrt(np.mean((y[mask] - y_rot[mask]) ** 2)))

# recompute the figures at the displayed resolution (clobbered by the loop)
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")
y = np.exp(-(x1**2 + x2**2))
y_rot = scipy.ndimage.rotate(y, ANGLE, reshape=False)
vmin, vmax = np.min(y), np.max(y)

if not args.book:
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(y.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
    axs[1].imshow(y_rot.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
    for ax in axs:
        ax.axis("off")

    fig, ax = plt.subplots()
    ax.plot(resolutions, maes, "ko")
    # ax.plot(resolutions, rmses, "r")
    ax.set_yscale("log")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.imshow(y.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / f"pixelized_gaussian_{RESOLUTION}.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.imshow(y_rot.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / f"pixelized_gaussian_rot_{RESOLUTION}.png")
    plt.close()

    save_csv(RESULTS_DIR / "pixelized_gaussian_mae.csv", x=resolutions, y=maes)
