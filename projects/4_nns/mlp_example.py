import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 256
EPOCHS = 200
LR = 2e-2

# define loss
cost_fun = nn.MSELoss()

# model settings
LAYERS = [8, 32, 32, 32, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
x_ = np.linspace(-1, 1, RESOLUTION)
y = np.sin(8 * np.pi * x_) * np.sin(6 * np.pi * x_)
y = torch.from_numpy(y).to(torch.float32).unsqueeze(1).to(device)

# random noise mapped to the target
x = torch.randn((RESOLUTION, LAYERS[0]), dtype=torch.float32).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS)
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
    ax.plot(x_, y_pred.detach().cpu(), "k")
    ax.plot(x_, y.detach().cpu(), "r--")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.plot(x_, y_pred.detach().cpu(), "k", linewidth=1.5)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / "MLP_prediction.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.plot(x_, y.detach().cpu(), "k", linewidth=1.5)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / "MLP_target.pdf")
    plt.close()

    # stacked random-noise input channels
    fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
    for i in range(LAYERS[0]):
        ax.plot(x_ + 0.1 * i, x[:, i].detach().cpu() - 3 * i, "k", linewidth=1.5)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / "MLP_input.pdf")
    plt.close()
