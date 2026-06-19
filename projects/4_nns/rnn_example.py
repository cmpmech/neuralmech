import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import DRNN
from postprocessing import save_csv

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
EPOCHS = 3000
LR = 1e-3

# define loss
cost_fun = nn.MSELoss()

# model settings
CELL = nn.RNN
# CELL = nn.LSTM
# CELL = nn.GRU
LAYERS = [1, 16, 16, 1]

if CELL == nn.GRU:
    LR = 2e-3

# ------------------------------------ prepare data -----------------------------------
x_ = np.linspace(0, 1, RESOLUTION)
y_ = 0.5 * (np.sin(10 * np.pi * x_) + np.sin(23.7 * np.pi * x_))  # cannot be integer

# constant input triggered by an initial impulse
x = torch.ones(1, RESOLUTION, 1, dtype=torch.float32).to(device) * 0.5
x[:, 0, :] = 1.0
y = torch.from_numpy(y_).float().view(1, RESOLUTION, 1).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = DRNN(LAYERS, cell=CELL)
model.to(device)
init_weights(model, nn.Tanh())
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

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.plot(x_, y[0, :, 0].detach().cpu(), "k")
    ax.plot(x_, y_pred[0, :, 0].detach().cpu(), "r--")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    cell2string = {nn.RNN: "rnn", nn.LSTM: "lstm", nn.GRU: "gru"}
    save_csv(
        RESULTS_DIR / f"rnn_sine_{cell2string[CELL]}.csv",
        i=np.arange(RESOLUTION) + 1,
        x=x.squeeze().detach().cpu(),
        z=x_,
        y=y.squeeze().detach().cpu(),
        y_pred=y_pred.squeeze().detach().cpu(),
    )
