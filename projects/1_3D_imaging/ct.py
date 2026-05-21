import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ----------------------- settings -----------------------
N = 600
HOLE_OFFSET_X = 0.22
HOLE_OFFSET_Y = 0.18
HOLE_RADIUS = 0.12
N_ANGLES = 30  # 90  # 180

# ----------------------- phantom ------------------------
coords = np.linspace(-1.0, 1.0, N)
x, y = np.meshgrid(coords, coords, indexing="ij")

theta = np.arctan2(y, x)
r_grid = np.sqrt(x**2 + y**2)

# potato: smooth closed boundary via low-frequency polar Fourier series
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

# -------------------- radon transform -------------------
phi = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)
sinogram = radon(phantom, theta=phi)
s = np.linspace(-N / 2, N / 2, sinogram.shape[0])


# -------------- filtered backprojection (FBP) -----------
reconstruction = iradon(sinogram, theta=phi, filter_name="ramp")
# reconstruction = iradon(sinogram, theta=phi, filter_name=None)
# reconstruction = iradon(sinogram, theta=phi, filter_name="shepp-logan")

# ------------------- post-processing --------------------
fig1, ax1 = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
ax1.imshow(phantom.T, origin="lower", cmap="binary")
ax1.axis("off")
ax1.set_rasterized(True)
fig1.tight_layout(pad=0)
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
fig.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / "ct_sinogram.png")
plt.show()

fig1, ax1 = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
ax1.imshow(reconstruction.T, origin="lower", cmap="binary")
ax1.axis("off")
ax1.set_rasterized(True)
fig1.tight_layout(pad=0)
plt.savefig(RESULTS_DIR / "ct_prediction.png")
plt.show()


# fig1, ax1 = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
# ax1.imshow(phantom.T, origin="lower", cmap="binary_r", extent=[-1, 1, -1, 1])
# ax1.axis("off")
# ax1.set_rasterized(True)
# fig1.tight_layout(pad=0)

# fig2, ax2 = plt.subplots(figsize=(N_ANGLES / 10, N / 10), dpi=100)
# ax2.imshow(
#     sinogram,
#     aspect="auto",
#     cmap="gray",
#     extent=[phi[0], phi[-1], s[-1], s[0]],
# )
# ax2.axis("off")
# ax2.set_rasterized(True)
# fig2.tight_layout(pad=0)

# fig3, ax3 = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
# ax3.imshow(reconstruction.T, origin="lower", cmap="binary_r")
# ax3.axis("off")
# ax3.set_rasterized(True)
# fig3.tight_layout(pad=0)

# if args.book:
#     fig1.savefig(RESULTS_DIR / "ct_phantom.png")
#     fig2.savefig(RESULTS_DIR / "ct_sinogram.png")
#     fig3.savefig(RESULTS_DIR / "ct_reconstruction.png")
#     plt.close("all")
# else:
#     plt.show()
