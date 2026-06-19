import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP, NODE

BASE_DIR = Path(__file__).parent

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small
rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
SAMPLES = 32
EPOCHS = 100
LR = 2e-2

# define loss
cost_fun = nn.MSELoss()

# model settings
# augmented state has two channels; only the first is fit to the data
LAYERS = [3, 32, 32, 2]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
x_ = np.sort(rng.uniform(0, 1, SAMPLES))
x_[0] = 0  # include 0 as initial condition
y_ = np.sin(2 * np.pi * x_)

x = torch.from_numpy(x_).float().to(device)
y = torch.from_numpy(y_).float().view(SAMPLES, 1, 1).to(device)  # (seq len, batch, features)
y0 = y[0].repeat((1, 2))  # inflate the initial condition to the augmented state

# --------------------------- instantiate model & optimizer ---------------------------
rhs_model = MLP(LAYERS, ACTIVATIONS)
model = NODE(rhs_model)
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
    y_pred = model(x, y0)[:, :, 0:1]  # match only the first channel to the data
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------------- prediction ------------------------------------
x = torch.linspace(0, 1, 256, dtype=torch.float32).to(device)
y0 = torch.zeros((1, 2), dtype=torch.float32).to(device)
y_pred = model(x, y0)[:, :, 0:1]

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(train_cost, "k")
plt.show()

fig, ax = plt.subplots()
ax.plot(x_, y_, "ko")
ax.plot(x.cpu(), y_pred[:, 0, 0].detach().cpu(), "r--")
plt.show()
