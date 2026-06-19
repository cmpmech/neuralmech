import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
N = 600
HOLE_OFFSET_X = 0.22
HOLE_OFFSET_Y = 0.18
HOLE_RADIUS = 0.12
ANGLES = 30  # 90  # 180 # more angles improve the reconstruction

# ---------------------------------- phantom geometry ---------------------------------
coords = np.linspace(-1.0, 1.0, N)
x, y = np.meshgrid(coords, coords, indexing="ij")

theta = np.arctan2(y, x)
r_grid = np.sqrt(x**2 + y**2)

# potato geometry (smooth closed boundary via low-frequency polar Fourier series)
r_boundary = (
    0.4
    + 0.12 * np.cos(theta)
    + 0.07 * np.cos(2 * theta)
    + 0.04 * np.cos(3 * theta)
    + 0.09 * np.sin(theta)
    + 0.05 * np.sin(2 * theta)
)

phantom = np.zeros((N, N))
phantom[r_grid <= r_boundary] = 1.0
phantom[(x - HOLE_OFFSET_X) ** 2 + (y - HOLE_OFFSET_Y) ** 2 <= HOLE_RADIUS**2] = 0.0

# ---------------------------------- radon transform ----------------------------------
phi = np.linspace(0.0, 180.0, ANGLES, endpoint=False)
sinogram = radon(phantom, theta=phi)  # measurements
s = np.linspace(-N / 2, N / 2, sinogram.shape[0])

# ------------------------------ filtered backprojection ------------------------------
reconstruction = iradon(sinogram, theta=phi, filter_name="ramp")
# reconstruction = iradon(sinogram, theta=phi, filter_name=None)
# reconstruction = iradon(sinogram, theta=phi, filter_name="shepp-logan")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
ax.imshow(phantom.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "ct_original.png")
plt.show()

angle = 60
measurement = radon(phantom, theta=[angle])[:, 0]
save_csv(
    RESULTS_DIR / "ct_signal.csv",
    x=s,
    y=measurement,
)

fig, ax = plt.subplots()
ax.imshow(sinogram, aspect="auto", cmap="binary_r")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "ct_sinogram.png")
plt.show()

fig, ax = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
ax.imshow(reconstruction.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "ct_prediction.png")
plt.show()
