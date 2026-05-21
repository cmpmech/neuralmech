import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from NN import DCN, UNet

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
DOMAIN_SIZE = 128
N_EXAMPLES = 7

# ----------------------------- data -----------------------------
data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fibers_test_{DOMAIN_SIZE}.npy")
)
masks = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fiber_masks_test_{DOMAIN_SIZE}.npy")
)
data = data.to(torch.float32)
masks = masks.to(torch.float32)

x_gt = data[:N_EXAMPLES].unsqueeze(1).to(device)  # (N, 1, H, W)
masks = masks[:N_EXAMPLES].unsqueeze(1).to(device)  # (N, 1, H, W)
b = x_gt * masks


# ---------------------- operators -------------------------------
def A_op(x, mask):
    return mask * x


def At_op(y, mask):
    return mask * y


def cg_solve(mask, rhs, lam, n_iter=10):
    def Lx(v):
        return At_op(A_op(v, mask), mask) + lam * v

    x = torch.zeros_like(rhs)
    r = rhs - Lx(x)
    p = r.clone()
    rs_old = (r * r).sum()
    for _ in range(n_iter):
        Ap = Lx(p)
        alpha = rs_old / ((p * Ap).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < 1e-6:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    return x


class MoDL(nn.Module):
    def __init__(self, denoiser, K, lam, cg_iter):
        super().__init__()
        self.K = K
        self.lam = lam
        self.cg_iter = cg_iter
        self.D_w = denoiser

    def forward(self, b, mask):
        x = b.clone()
        for _ in range(self.K):
            z = self.D_w(x)
            rhs = At_op(b, mask) + self.lam * z
            x = cg_solve(mask, rhs, self.lam, self.cg_iter)
        return x


# --------------------------- model ------------------------------
model = torch.load(
    BASE_DIR / f"../../models/modl_{DOMAIN_SIZE}.pt2",
    map_location=device,
    weights_only=False,
)
model.eval()

# -------------------- reconstruction ----------------------------
with torch.no_grad():
    x_rec = model(b, masks)

# ----------------------- postprocessing -------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for i in range(N_EXAMPLES):
    fig, ax = plt.subplots(figsize=(DOMAIN_SIZE / 100, DOMAIN_SIZE / 100), dpi=100)
    ax.imshow(x_rec[i, 0].T.cpu(), origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(RESULTS_DIR / f"img_prediction_modl_{i}.png")
    plt.close()
