from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from DL import init_weights
from helper import MoDL
from NN import DCN, UNet

BASE_DIR = Path(__file__).parent
TMP_DIR = (BASE_DIR / "../../tmp").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 1000
LR = 2e-3
BATCH_SIZE = 16
K = 10  # 15
LAM = 0.2  # 0.5
CG_ITER = 10
MASK_RATIO = 0.6
BASE_CH = 32
DEPTH = 3
DOMAIN_SIZE = 128
PRINT_EVERY = 10
OVERFIT = None  # set to int for quick overfit test

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# ------------------------------------ prepare data -----------------------------------
data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fibers_{DOMAIN_SIZE}.npy")
)
data = data.to(torch.float32).unsqueeze(1)  # (N, 1, H, W)
if OVERFIT is not None:
    data = data[:OVERFIT]


class FiberMaskDataset(Dataset):
    def __init__(self, tensors, mask_ratio=0.5):
        self.data = tensors
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_gt = self.data[idx]
        if torch.rand(1).item() > 0.5:
            x_gt = x_gt.flip(-1)
        if torch.rand(1).item() > 0.5:
            x_gt = x_gt.flip(-2)
        mask = (torch.rand_like(x_gt) > self.mask_ratio).float()
        return x_gt * mask, mask, x_gt


dataset = FiberMaskDataset(data, MASK_RATIO)
n_val = max(1, int(0.1 * len(dataset)))
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_set, batch_size=len(val_set))

# TODO standardization?

# --------------------------- instantiate model & optimizer ---------------------------
levels = [1] + [BASE_CH * 2**i for i in range(DEPTH)]  # [1, 16, 32, 64]
# act = nn.ReLU
act = nn.GELU

downs = [
    DCN(
        [levels[i], levels[i + 1], levels[i + 1]],
        [[nn.GroupNorm(1, levels[i + 1]), act()] for _ in range(2)],
        3,
        stride=[1, 2],
        padding=1,
        dim=2,
    )
    for i in range(DEPTH)
]

ups = [
    DCN(
        [2 * levels[i + 1], levels[i + 1], levels[i] if i > 0 else 1],
        [
            [nn.GroupNorm(1, levels[i + 1]), act()],
            [nn.GroupNorm(1, levels[i]), act()] if i > 0 else None,
        ],
        3,
        stride=1,
        padding=1,
        dim=2,
        pre_modules=[
            nn.Upsample(scale_factor=2, mode="nearest"),
            None,
        ],
    )
    for i in reversed(range(DEPTH))
]

denoiser = nn.Sequential(UNet(downs, ups), nn.Sigmoid())

model = MoDL(denoiser, K=K, lam=LAM, cg_iter=CG_ITER).to(device)
init_weights(model.D_w, act())
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
)

# -------------------------------------- training -------------------------------------
train_cost = [0.0] * EPOCHS
val_cost = [0.0] * EPOCHS
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for b, mask, x_gt in train_loader:
        b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
        optimizer.zero_grad()
        x_pred = model(b, mask)
        cost = cost_fun(x_pred, x_gt)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
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

# ------------------------------------- save model ------------------------------------
torch.save(model, BASE_DIR / f"../../models/modl_{DOMAIN_SIZE}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.savefig(TMP_DIR / "hist.png")
plt.close()

model.eval()
b, mask, x_gt = next(iter(val_loader))
b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
with torch.no_grad():
    x_pred = model(b, mask)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
for ax, img in zip(axes, [b, x_pred, x_gt]):
    ax.imshow(img[0, 0].cpu(), cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(TMP_DIR / "recon.png")
plt.close()

model.eval()
b, mask, x_gt = next(iter(train_loader))
b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
with torch.no_grad():
    x_pred = model(b, mask)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
for ax, img in zip(axes, [b, x_pred, x_gt]):
    ax.imshow(img[0, 0].cpu(), cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(TMP_DIR / "recon_train.png")
plt.close()
