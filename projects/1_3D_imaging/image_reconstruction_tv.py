import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from helper import cg
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
# TODO could be animated
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 128
EXAMPLES = 7
USE_TV = False  # True
LAMBDA = 0.1  # TV weight
RHO = 0.1  # ADMM penalty parameter
ADMM_ITER = 200
CG_ITER = 20

# ------------------------------------- load data -------------------------------------
data = torch.from_numpy(np.load(DATA_DIR / f"graded_fibers_test_{RESOLUTION}.npy"))
masks = torch.from_numpy(
    np.load(DATA_DIR / f"graded_fiber_masks_test_{RESOLUTION}.npy")
)
data = data.to(torch.float32)
masks = masks.to(torch.int)

x_gt = data[:EXAMPLES].to(device)  # (N, H, W)
masks = masks[:EXAMPLES].to(device)
b = x_gt * masks  # masked observations


# --------------------------------------- helper --------------------------------------
# for gradient
def lap(x):
    # D^T D x, discrete Laplacian with periodic BC
    return (
        4 * x
        - torch.roll(x, 1, -1)
        - torch.roll(x, -1, -1)
        - torch.roll(x, 1, -2)
        - torch.roll(x, -1, -2)
    )


def fdiff(x):
    # forward finite differences; returns (dh, dv): (..., H, W)
    return torch.roll(x, -1, -1) - x, torch.roll(x, -1, -2) - x


def fdiv(zh, zv):
    # adjoint of fdiff: D^T (zh, zv)
    return (torch.roll(zh, 1, -1) - zh) + (torch.roll(zv, 1, -2) - zv)


def soft_thresh(x, t):
    return x.sign() * (x.abs() - t).clamp(min=0.0)


# ----------------------------------- reconstruction ----------------------------------
def reconstruct(b_i, mask_i):  # b_i (measurement), mask_i (mask): (H, W)
    if not USE_TV:
        return b_i.clone()  # (trivial) pseudoinverse for masking = zero-fill

    def A_fn(v):  # v is reconstruction
        return mask_i * v + RHO * lap(v)  # mask^2 = mask for binary mask

    # ADMM split: x-update (CG) + TV-prox (soft-threshold) + dual update (see README)
    x = b_i.clone()
    zh, zv = fdiff(x)  # horizontal & vertical gradients
    zh, zv = zh.clone(), zv.clone()
    uh = torch.zeros_like(zh)
    uv = torch.zeros_like(zv)

    # solved with augmented Lagrangian using ADMM
    for _ in range(ADMM_ITER):
        rhs = mask_i * b_i + RHO * fdiv(zh - uh, zv - uv)
        x = cg(A_fn, rhs, CG_ITER, tol=1e-7)

        dh, dv = fdiff(x)
        zh = soft_thresh(dh + uh, LAMBDA / RHO)
        zv = soft_thresh(dv + uv, LAMBDA / RHO)
        uh = uh + dh - zh
        uv = uv + dv - zv

    return x.clamp(0.0, 1.0)


x_rec = torch.stack([reconstruct(b[i], masks[i]) for i in tqdm(range(EXAMPLES))])

# ----------------------------------- postprocessing ----------------------------------
for i in range(EXAMPLES):
    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.imshow(x_gt[i].T.cpu(), origin="lower", cmap="binary")
    ax.imshow(
        (1 - masks[i]).T.cpu(),
        origin="lower",
        cmap="viridis_r",
        alpha=(1 - masks[i].cpu()).T.float(),
    )
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PDF_DIR / f"img_measurement_{i}.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.imshow(x_gt[i].T.cpu(), origin="lower", cmap="binary")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PDF_DIR / f"img_groundtruth_{i}.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.imshow(x_rec[i].T.cpu(), origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PDF_DIR / f"img_prediction_tv_{i}_{USE_TV}.pdf")
    plt.close()
