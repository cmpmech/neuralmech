import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import iradon
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ----------------------- settings -----------------------
N = 300
D = 3.0  # source-to-center distance (normalized, circle radius = 1)
N_BETA = 360  # source rotation positions (full circle)
N_DET = 400  # detector elements
N_STEPS = 200  # integration steps per ray
HOLE_OFFSET_X = 0.22
HOLE_OFFSET_Y = 0.18
HOLE_RADIUS = 0.12

# ----------------------- phantom ------------------------
coords = np.linspace(-1.0, 1.0, N)
x, y = np.meshgrid(coords, coords, indexing="ij")

theta_p = np.arctan2(y, x)
r_grid = np.sqrt(x**2 + y**2)

r_boundary = (
    0.4
    + 0.12 * np.cos(theta_p)
    + 0.07 * np.cos(2 * theta_p)
    + 0.04 * np.cos(3 * theta_p)
    + 0.09 * np.sin(theta_p)
    + 0.05 * np.sin(2 * theta_p)
)

phantom = np.zeros((N, N))
phantom[r_grid <= r_boundary] = 1.0
phantom[(x - HOLE_OFFSET_X) ** 2 + (y - HOLE_OFFSET_Y) ** 2 <= HOLE_RADIUS**2] = 0.0

# ------------------- fan beam forward -------------------
# source at angle beta, ray fan spanning [-gamma_max, gamma_max]
# ray direction at fan angle gamma: d = (-cos(beta+gamma), -sin(beta+gamma))
# intersection with unit circle: lambda = D*cos(gamma) +/- sqrt(1 - D^2*sin^2(gamma))
R = 1.0
gamma_max = np.arcsin(R / D)
gamma = np.linspace(-gamma_max, gamma_max, N_DET)
beta = np.linspace(0.0, 2.0 * np.pi, N_BETA, endpoint=False)

cos_g = np.cos(gamma)
sin_g = np.sin(gamma)
lam_c = D * cos_g
lam_r = np.sqrt(R**2 - (D * sin_g) ** 2)
lam_enter = lam_c - lam_r
lam_exit = lam_c + lam_r
dl = (lam_exit - lam_enter) / N_STEPS  # arc length per step per detector element

lam_unit = np.linspace(0.0, 1.0, N_STEPS)
sinogram_fan = np.zeros((N_DET, N_BETA))

for i, b in enumerate(tqdm(beta, desc="fan forward")):
    src_x = D * np.cos(b)
    src_y = D * np.sin(b)
    d_x = -np.cos(b + gamma)  # (N_DET,)
    d_y = -np.sin(b + gamma)

    # sample points along each ray: (N_DET, N_STEPS)
    lam_vals = lam_enter[:, None] + lam_unit[None, :] * (lam_exit - lam_enter)[:, None]
    pts_x = src_x + lam_vals * d_x[:, None]
    pts_y = src_y + lam_vals * d_y[:, None]

    # convert physical coords to phantom pixel indices
    px = (pts_x + 1.0) / 2.0 * N - 0.5
    py = (pts_y + 1.0) / 2.0 * N - 0.5

    vals = map_coordinates(
        phantom, [px.ravel(), py.ravel()], order=1, mode="constant", cval=0.0
    ).reshape(N_DET, N_STEPS)

    sinogram_fan[:, i] = np.sum(vals, axis=1) * dl

# -------------- rebinning: fan → parallel ---------------
# geometry: s = D*sin(gamma), phi = beta + gamma + pi/2 (mod pi)
# inverse: gamma = arcsin(s/D), beta = phi - gamma - pi/2 (mod 2pi)
N_PAR_S = N_DET
N_PAR_PHI = 180

s_par = np.linspace(-R, R, N_PAR_S)
phi_par = np.linspace(0.0, np.pi, N_PAR_PHI, endpoint=False)

gamma_rebin = np.arcsin(np.clip(s_par / D, -1.0, 1.0))  # (N_PAR_S,)
beta_rebin = (phi_par[:, None] - gamma_rebin[None, :] - np.pi / 2) % (2.0 * np.pi)  # (N_PAR_PHI, N_PAR_S)

gamma_idx = (gamma_rebin - gamma[0]) / (gamma[-1] - gamma[0]) * (N_DET - 1)
beta_idx = beta_rebin / (2.0 * np.pi) * N_BETA

sinogram_par = map_coordinates(
    sinogram_fan,
    [np.tile(gamma_idx, (N_PAR_PHI, 1)).ravel(), beta_idx.ravel()],
    order=1,
    mode="wrap",
).reshape(N_PAR_PHI, N_PAR_S).T  # (N_PAR_S, N_PAR_PHI) for iradon

# -------------- filtered backprojection (FBP) -----------
reconstruction = iradon(sinogram_par, theta=np.degrees(phi_par), filter_name="ramp")

# ------------------- post-processing --------------------
fig1, ax1 = plt.subplots(figsize=(N / 10, N / 10), dpi=100)
ax1.imshow(phantom.T, origin="lower", cmap="binary_r", extent=[-1, 1, -1, 1])
ax1.axis("off")
ax1.set_rasterized(True)
fig1.tight_layout(pad=0)

fig2, ax2 = plt.subplots(figsize=(N_BETA / 10, N_DET / 10), dpi=100)
ax2.imshow(sinogram_fan, aspect="auto", cmap="gray")
ax2.axis("off")
ax2.set_rasterized(True)
fig2.tight_layout(pad=0)

rec_n = reconstruction.shape[0]
fig3, ax3 = plt.subplots(figsize=(rec_n / 10, rec_n / 10), dpi=100)
ax3.imshow(reconstruction.T, origin="lower", cmap="binary_r")
ax3.axis("off")
ax3.set_rasterized(True)
fig3.tight_layout(pad=0)

if args.book:
    fig1.savefig(RESULTS_DIR / "ct_v2_phantom.png")
    fig2.savefig(RESULTS_DIR / "ct_v2_sinogram.png")
    fig3.savefig(RESULTS_DIR / "ct_v2_reconstruction.png")
    plt.close("all")
else:
    plt.show()
