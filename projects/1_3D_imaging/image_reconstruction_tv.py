import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ----------------------- hyperparameters ------------------------
DOMAIN_SIZE = 128  # 256  # 128
N_EXAMPLES = 7  # TODO where is this?
USE_TV = False  # True  # False  # True  # False  # True  # False: zero-fill (min-norm), True: TV-regularized ADMM
LAM = 0.1  # 0.05  # 0.02  # TV weight
RHO = 0.1  # ADMM penalty parameter
ADMM_ITER = 200
CG_ITER = 20

# ----------------------------- data -----------------------------
data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fibers_test_{DOMAIN_SIZE}.npy")
)
masks = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fiber_masks_test_{DOMAIN_SIZE}.npy")
)
data = data.to(torch.float32)
masks = masks.to(torch.int)

x_gt = data[:N_EXAMPLES].to(device)  # (N, H, W)
masks = masks[:N_EXAMPLES].to(device)
b = x_gt * masks  # masked observations


# ---------------------- operators -------------------------------
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


def cg(A_fn, rhs, n_iter):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum()
    for _ in range(n_iter):
        Ap = A_fn(p)
        alpha = rs / ((p * Ap).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < 1e-7:
            break
        p = r + (rs_new / (rs + 1e-12)) * p
        rs = rs_new
    return x


# -------------------- reconstruction ----------------------------
def reconstruct(b_i, mask_i):
    # b_i, mask_i: (H, W)
    if not USE_TV:
        return b_i.clone()  # pseudoinverse for masking = zero-fill

    def A_fn(v):
        return mask_i * v + RHO * lap(v)  # mask^2 = mask for binary mask

    x = b_i.clone()
    zh, zv = fdiff(x)
    zh, zv = zh.clone(), zv.clone()
    uh = torch.zeros_like(zh)
    uv = torch.zeros_like(zv)

    for _ in range(ADMM_ITER):
        rhs = mask_i * b_i + RHO * fdiv(zh - uh, zv - uv)
        x = cg(A_fn, rhs, CG_ITER)

        dh, dv = fdiff(x)
        zh = soft_thresh(dh + uh, LAM / RHO)
        zv = soft_thresh(dv + uv, LAM / RHO)
        uh = uh + dh - zh
        uv = uv + dv - zv

    return x.clamp(0.0, 1.0)


x_rec = torch.stack(
    [
        reconstruct(b[i], masks[i])
        for i in tqdm(range(N_EXAMPLES), desc="Reconstructing")
    ]
)

# ----------------------- postprocessing -------------------------
for i in range(N_EXAMPLES):
    fig, ax = plt.subplots(figsize=(DOMAIN_SIZE / 100, DOMAIN_SIZE / 100), dpi=100)
    ax.imshow(x_gt[i].T.cpu(), origin="lower", cmap="binary")
    ax.imshow(
        (1 - masks[i]).T.cpu(),
        origin="lower",
        cmap="viridis_r",
        alpha=(1 - masks[i].cpu()).T.float(),
    )
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(RESULTS_DIR / f"img_measurement_{i}.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(DOMAIN_SIZE / 100, DOMAIN_SIZE / 100), dpi=100)
    ax.imshow(x_gt[i].T.cpu(), origin="lower", cmap="binary")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(RESULTS_DIR / f"img_groundtruth_{i}.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(DOMAIN_SIZE / 100, DOMAIN_SIZE / 100), dpi=100)
    ax.imshow(x_rec[i].T.cpu(), origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(RESULTS_DIR / f"img_prediction_tv_{i}_{USE_TV}.png")
    plt.close()
