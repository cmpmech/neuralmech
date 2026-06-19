import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import DCN

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
RESOLUTION = 256
EPOCHS = 2000
LR = 2e-2

# define loss
cost_fun = nn.MSELoss()

# model settings
CHANNELS = [16, 16, 16, 16, 1]
KERNEL_SIZE, STRIDE, PADDING = 3, 1, 1
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(CHANNELS) - 2)]
# group norm with one group is layer norm without specifying the image size
NORMALIZATIONS = [nn.GroupNorm(1, CHANNELS[i + 1]) for i in range(len(CHANNELS) - 2)]
RESAMPLINGS = [
    nn.Upsample(scale_factor=2, mode="bilinear") for _ in range(len(CHANNELS) - 2)
]

# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

y = np.sin(2 * np.pi * x1) * np.sin(12 * np.pi * x1 * x2)
y = torch.from_numpy(y).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)

# random noise mapped to the target, upsampled back to full resolution by the network
input_size = RESOLUTION // 2 ** (len(CHANNELS) - 2)
x = torch.randn((1, CHANNELS[0], input_size, input_size), dtype=torch.float32).to(
    device
)

# --------------------------- instantiate model & optimizer ---------------------------
model = DCN(
    CHANNELS,
    ACTIVATIONS,
    KERNEL_SIZE,
    STRIDE,
    PADDING,
    normalizations=NORMALIZATIONS,
    resamplings=RESAMPLINGS,
)
model.to(device)
init_weights(model, ACTIVATIONS[0])
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
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

    fig, ax = plt.subplots()
    ax.pcolormesh(x1, x2, y_pred[0, 0].detach().cpu(), vmin=-1, vmax=1, cmap="Spectral")
    ax.set_aspect("equal")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:

    def save_field(field, path):
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
        ax.pcolormesh(x1, x2, field, vmin=-1, vmax=1, cmap="Spectral")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path)
        plt.close()

    save_field(y_pred[0, 0].detach().cpu(), RESULTS_DIR / "CNN_prediction.png")
    save_field(y[0, 0].detach().cpu(), RESULTS_DIR / "CNN_target.png")
    for i in range(CHANNELS[0]):
        fig, ax = plt.subplots(figsize=(input_size / 50, input_size / 50), dpi=100)
        ax.imshow(
            x[0, i].detach().cpu().T, origin="lower", vmin=-1, vmax=1, cmap="Spectral"
        )
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RESULTS_DIR / f"CNN_input_{i}.png")
        plt.close()
