from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import DCN, UNet

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# ----------------------- hyperparameters ------------------------
# EPOCHS = 200
# LR = 1e-3
# BATCH_SIZE = 8 #32  # 4
# K = 6
# LAM = 0.5
# CG_ITER = 10
# MASK_RATIO = 0.7  # 0.5
# BASE_CH = 16
# DEPTH = 3
# DOMAIN_SIZE = 128  # 256
# PRINT_EVERY = 1

# EPOCHS = 1000
# LR = 5e-3 #2e-3 #2e-3 #5e-3 #1e-3
# BATCH_SIZE = 9 #8 #32  # 4 # full batch currently
# K = 10
# LAM = 0.5
# CG_ITER = 10
# MASK_RATIO = 0.7  # 0.5
# BASE_CH = 32 #16
# DEPTH = 3
# DOMAIN_SIZE = 128  # 256
# PRINT_EVERY = 1
#
EPOCHS = 1000 # could be extended
# LR = 2e-3 #2e-3 #2e-3 #5e-3 #1e-3
LR = 2e-3 #2e-3 #2e-3 #5e-3 #1e-3
BATCH_SIZE = 1 #32 #16 #9 #8 #32  # 4 # full batch currently
K = 8 #10
LAM = 0.5
CG_ITER = 10
MASK_RATIO = 0.5 #0.8 # 0.5 #0.7  # 0.5
BASE_CH = 16 #32 #32 helps a little
DEPTH = 3
DOMAIN_SIZE = 128  # 256
PRINT_EVERY = 10

cost_fun = nn.MSELoss(reduction="mean")

# ----------------------------- data -----------------------------
data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fibers_{DOMAIN_SIZE}.npy")
)
data = data.to(torch.float32).unsqueeze(1)  # (N, 1, H, W)


class FiberMaskDataset(Dataset):
    def __init__(self, tensors, mask_ratio=0.5):
        self.data = tensors
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_gt = self.data[idx]
        mask = (torch.rand_like(x_gt) > self.mask_ratio).float()
        return x_gt * mask, mask, x_gt


dataset = FiberMaskDataset(data, MASK_RATIO)
train_set, val_set = random_split(dataset, [0.5, 0.5]) # TODO fix
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# TODO standardization?

# ------------------- forward operator ---------------------------
def A_op(x, mask):
    return mask * x


def At_op(y, mask):
    return mask * y  # A^T = A for masking


def cg_solve(mask, rhs, lam: float, n_iter: int = 10) -> torch.Tensor:
    # solves (A^T A + λI) x = rhs; for masking: (mask + λ) x = rhs
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


# --------------------------- model ------------------------------
levels = [1] + [BASE_CH * 2**i for i in range(DEPTH)]  # [1, 16, 32, 64]
# activations = [nn.GELU()]
act = nn.ReLU(inplace=True)
# act = nn.GELU()
# act = nn.LeakyReLU()
# act = nn.PReLU()

downs = [
    DCN(
        [levels[i], levels[i + 1], levels[i + 1]],
        [act, act],
        3,
        stride=[1, 2],
        padding=1,
        dim=2,
        normalizations=[nn.GroupNorm(1, levels[i + 1]) for _ in range(2)],
    )
    for i in range(DEPTH)
]

ups = [
    DCN(
        [2 * levels[i + 1], levels[i + 1], levels[i] if i > 0 else 1],
        [act, act if i > 0 else None],
        3,
        stride=1,
        padding=1,
        dim=2,
        resamplings=[
            nn.Upsample(scale_factor=2, mode="nearest"), # TODO try nearest instead of bilinear (with align_corners)
            None,
        ],
        normalizations=[
            nn.GroupNorm(1, levels[i + 1]),
            nn.GroupNorm(1, levels[i]) if i > 0 else None,
        ],
    )
    for i in reversed(range(DEPTH))
]

denoiser = UNet(downs, ups)
init_weights(denoiser, act)

class MoDL(nn.Module):
    # K unrolled iterations: z = D_w(x), x = CG-solve(A^TA + λI | A^Tb + λz)
    # D_w is weight-shared across all K iterations
    def __init__(self, denoiser: nn.Module, K: int, lam: float, cg_iter: int):
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


model = MoDL(denoiser, K=K, lam=LAM, cg_iter=CG_ITER).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
)
# scheduler = None

# -------------------------- training ----------------------------
train_cost = [0.0] * EPOCHS
val_cost = [0.0] * EPOCHS
pbar = tqdm(range(EPOCHS), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for b, mask, x_gt in train_loader:
        b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
        optimizer.zero_grad()
        x_pred = model(b, mask)
        cost = cost_fun(x_pred, x_gt)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for b, mask, x_gt in val_loader:
            b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
            val_cost[epoch] += cost_fun(model(b, mask), x_gt).item()
        val_cost[epoch] /= len(val_loader)

    if epoch % PRINT_EVERY == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

# --------------------------- save model -------------------------
torch.save(model, BASE_DIR / f"../../models/modl_{DOMAIN_SIZE}.pt2")

# ----------------------- postprocessing -------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.savefig('../../tmp/hist.png')
plt.show()

model.eval()
b, mask, x_gt = next(iter(val_loader))
b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
with torch.no_grad():
    x_pred = model(b, mask)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
for ax, img in zip(axes, [b, x_pred, x_gt]):
    ax.imshow(img[0, 0].cpu(), cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
fig.tight_layout(pad=0)
plt.savefig('../../tmp/recon.png')
plt.show()



model.eval()
b, mask, x_gt = next(iter(train_loader))
b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
with torch.no_grad():
    x_pred = model(b, mask)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
for ax, img in zip(axes, [b, x_pred, x_gt]):
    ax.imshow(img[0, 0].cpu(), cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
fig.tight_layout(pad=0)
plt.savefig('../../tmp/recon_train.png')
plt.show()
