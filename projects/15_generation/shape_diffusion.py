from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import DCN, UNet

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- training settings ---------------------------
epochs = 600
lr = 1e-3
weight_decay = 1e-2
batch_size = 64

T = 200  # diffusion timesteps
beta_start = 1e-4
beta_end = 0.02

cost_fun = nn.MSELoss(reduction="mean")

# ---------------------------- model settings ----------------------------
base, depth = 2, 4
channel_dim = 4
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

# ----------------------------- prepare data -----------------------------
domain_size = 128

labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]
data = []
for label in labels:
    data.append(
        torch.from_numpy(
            np.load(BASE_DIR / f"../../data/shapes_{label}_{domain_size}.npy")
        )
    )
data = torch.from_numpy(np.concatenate(data, axis=0))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))

# --------------------------- noise schedule -----------------------------
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# -------------------------- instantiate model ---------------------------
levels = [2] + [channel_dim * base ** (i + 1) for i in range(depth)]  # 2: x_t + t

downs = [
    DCN(
        [levels[i], levels[i + 1], levels[i + 1]],
        [act(), act()],
        kernel_size,
        stride=[1, 2],
        padding=kernel_size // 2,
        dim=2,
        normalizations=[nn.GroupNorm(1, levels[i + 1]) for _ in range(2)],
    )
    for i in range(depth)
]

ups = [
    DCN(
        [2 * levels[i + 1], levels[i + 1], levels[i] if i > 0 else 1],
        [act(), act() if i > 0 else None],
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        dim=2,
        resamplings=[
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            None,
        ],
        normalizations=[
            nn.GroupNorm(1, levels[i + 1]),
            nn.GroupNorm(1, levels[i]) if i > 0 else None,
        ],
    )
    for i in reversed(range(depth))
]

model = UNet(downs, ups).to(device)
init_weights(model, act())
summary(model, (1, 2, domain_size, domain_size), depth=4)

# ------------------------ instantiate optimizer -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)

# ------------------------------- training -------------------------------
print_every = 10
train_cost = [0] * epochs
val_cost = [0] * epochs
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x0 = standardizex(x[0]).to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        eps = torch.randn_like(x0)
        a_bar = alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps  # eq:diffusion_reparam
        t_chan = (t.float() / T).view(-1, 1, 1, 1).expand_as(x0)
        eps_pred = model(torch.cat([xt, t_chan], dim=1))
        cost = cost_fun(eps_pred, eps)  # eq:diffusion_loss
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x0 = standardizex(x[0]).to(device)
            t = torch.randint(0, T, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            a_bar = alpha_bars[t].view(-1, 1, 1, 1)
            xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
            t_chan = (t.float() / T).view(-1, 1, 1, 1).expand_as(x0)
            eps_pred = model(torch.cat([xt, t_chan], dim=1))
            cost = cost_fun(eps_pred, eps)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

# ----------------------------- export model -----------------------------
model.standardizer = standardizex  # just for saving
model.T = T
model.betas = betas
model.alphas = alphas
model.alpha_bars = alpha_bars
torch.save(model, BASE_DIR / f"../../models/shape_diffusion_{T}_{domain_size}.pt2")

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.show()
