import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

BASE_DIR = Path(__file__).parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ----------------------- hyperparameters ------------------------
MASK_RATIO = 0.6
DOMAIN_SIZE = 128 #256  # 128
N_EXAMPLES = 8  # TODO where is this?
USE_TV = True  # False  # True  # False: zero-fill (min-norm), True: TV-regularized ADMM
SIGMA = 0  # Gaussian blur std (pixels) applied to binary data before masking;
# converts hard binary circles into smooth density maps where TV is meaningful
LAM = 0.05 #0.02  # TV weight
RHO = 0.1  # ADMM penalty parameter
ADMM_ITER = 200
CG_ITER = 20

# ----------------------------- data -----------------------------
data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fibers_{DOMAIN_SIZE}.npy")
)  # TODO graded_ ???
data = data.to(torch.float32)  # (N, H, W)
if SIGMA > 0:
    data = torch.from_numpy(gaussian_filter(data.numpy(), sigma=[0, SIGMA, SIGMA])).to(
        torch.float32
    )
    data = data / data.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

rng = torch.Generator()
rng.manual_seed(0)
idx = torch.randperm(len(data), generator=rng)[:N_EXAMPLES]
x_gt = data[idx].to(device)  # (N, H, W)
masks = (torch.rand(x_gt.shape, generator=rng) > MASK_RATIO).float().to(device)
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
mse = ((x_rec - x_gt) ** 2).mean(dim=(-2, -1))
print(f"Mean MSE: {mse.mean():.2e}")

n_show = min(4, N_EXAMPLES)
fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9), dpi=150)
for j in range(n_show):
    for ax, img in zip(axes[:, j], [b[j], x_rec[j], x_gt[j]]):
        ax.imshow(img.cpu(), cmap="binary", vmin=0, vmax=1)
        ax.axis("off")
fig.tight_layout(pad=0)

if args.book or args.animate:
    RESULTS_DIR = BASE_DIR / "../../results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "tv" if USE_TV else "noreg"
    fig.savefig(
        RESULTS_DIR / f"image_reconstruction_tv_{suffix}.pdf", bbox_inches="tight"
    )
    plt.close()
else:
    plt.savefig("../../tmp/tv_recon.png")
    plt.show()
