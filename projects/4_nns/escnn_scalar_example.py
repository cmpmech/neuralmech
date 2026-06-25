import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from escnn import gspaces
from escnn import nn as enn
from torch import nn
from tqdm import tqdm

from NN import EquivariantCNN

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 128
EPOCHS = 1000
LR = 2e-3

# define loss
cost_fun = nn.MSELoss()

# model settings
ROTATIONS = 1  # cnn recovered
# ROTATIONS = 8  # discrete C8 rotation group acting on R^2 (cheaper than full SO(2))

CHANNELS = [1] + [64 // ROTATIONS] * 4 + [1]
KERNEL_SIZE = 7
PADDING = KERNEL_SIZE // 2
ACTIVATION = enn.LeakyReLU

# postprocessing
ANGLES = [0, 30, 45, 90]

# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

x_np = np.sin(4 * np.pi * x1 * x2)
y_np = np.cos(4 * np.pi * x1 * x2)
x = torch.from_numpy(x_np).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
y = torch.from_numpy(y_np).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)

# only within the inscribed disk is the square grid actually rotation-equivariant
mask = torch.from_numpy((x1**2 + x2**2) <= 1.0).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
gspace = gspaces.rot2dOnR2(N=ROTATIONS)
model = EquivariantCNN(
    gspace, CHANNELS, ACTIVATION, kernel_size=KERNEL_SIZE, padding=PADDING
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x)
    cost = cost_fun(y_pred[0, 0][mask], y[0, 0][mask])
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
mask_np = mask.cpu().numpy()
model.eval()

targets, preds = {}, {}
for angle in ANGLES:
    x_rot = scipy.ndimage.rotate(x_np, angle, reshape=False)
    x_rot = torch.from_numpy(x_rot).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(x_rot)[0, 0].cpu().numpy()
    target = scipy.ndimage.rotate(y_np, angle, reshape=False)
    targets[angle] = np.where(mask_np, target, np.nan)
    preds[angle] = np.where(mask_np, pred, np.nan)

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    for angle in ANGLES:
        for field in (targets[angle], preds[angle]):
            fig, ax = plt.subplots()
            ax.pcolormesh(x1, x2, field, vmin=-1, vmax=1, cmap="Spectral")
            ax.set_aspect("equal")
            ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for angle in ANGLES:
        for name, field in (("target", targets[angle]), ("prediction", preds[angle])):
            fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
            ax.pcolormesh(x1, x2, field, vmin=-1, vmax=1, cmap="Spectral")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_rasterized(True)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(RESULTS_DIR / f"ESCNN_{name}_{angle}_{ROTATIONS}.png")
            plt.close()
