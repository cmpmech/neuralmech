import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from helper import UNet

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 16

# define loss
cost_fun = nn.BCEWithLogitsLoss()

# model settings
CHANNELS = [16, 32, 64, 128]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "neuralfwi.npz")
X = torch.tensor(data["gradients"], dtype=torch.float32)[:, None].to(device)
Y = torch.tensor(~data["voids"], dtype=torch.float32)[:, None].to(device)
split = int(0.9 * len(X))
X_train, Y_train, X_val, Y_val = X[:split], Y[:split], X[split:], Y[split:]

# --------------------------- instantiate model & optimizer ---------------------------
model = UNet(CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# -------------------------------------- training -------------------------------------
train_cost, val_cost = [], []
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    epoch_cost = 0.0
    batches = torch.randperm(split, device=device).split(BATCH_SIZE)
    for ids in batches:
        x, y = X_train[ids], Y_train[ids]
        if torch.rand(1) < 0.5:  # the transducer array is symmetric about x = 0.5
            x, y = x.flip(2), y.flip(2)
        optimizer.zero_grad()
        cost = cost_fun(model(x), y)
        cost.backward()
        optimizer.step()
        epoch_cost += cost.item()
    scheduler.step()
    train_cost.append(epoch_cost / len(batches))

    model.eval()
    with torch.no_grad():
        val_batches = zip(X_val.split(BATCH_SIZE), Y_val.split(BATCH_SIZE))
        costs = [cost_fun(model(x), y).item() for x, y in val_batches]
    val_cost.append(np.mean(costs))
    pbar.set_postfix(train=f"{train_cost[-1]:.2e}", val=f"{val_cost[-1]:.2e}")
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# --------------------------------------- export --------------------------------------
torch.save(model.state_dict(), MODEL_DIR / "neuralfwi_unet.pt")
print(f"saved to {MODEL_DIR / 'neuralfwi_unet.pt'}")

# ----------------------------------- postprocessing ----------------------------------
with torch.no_grad():
    y_pred = torch.sigmoid(model(X_val[:4])).cpu()
fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for i in range(4):
    gradient = X_val[i, 0].cpu().T
    axes[0, i].imshow(gradient, origin="lower", cmap="seismic", vmin=-0.3, vmax=0.3)
    axes[1, i].imshow(Y_val[i, 0].cpu().T, origin="lower", cmap="hot", vmin=0, vmax=1)
    axes[2, i].imshow(y_pred[i, 0].T, origin="lower", cmap="hot", vmin=0, vmax=1)
for ax in axes.flat:
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "neuralfwi_pretraining.png")
plt.close()
fig, ax = plt.subplots()
ax.semilogy(train_cost, "k")
ax.semilogy(val_cost, "r")
plt.savefig(RESULTS_DIR / "neuralfwi_pretraining_history.png")
