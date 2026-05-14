import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import DCN
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------- training settings ------------------------
RESOLUTION = 256  # 32
EPOCHS = 200
LR = 1e-2

cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
KERNEL_SIZE = 3

# RECEPTIVE_FIELD = 63
# RECEPTIVE_FIELD = 127
RECEPTIVE_FIELD = 255

if RECEPTIVE_FIELD == 255:
    CHANNELS = [1, 16, 16, 16, 16, 16, 16, 1]
    DILATIONS = [1, 2, 4, 8, 16, 32, 64]
elif RECEPTIVE_FIELD == 127:
    CHANNELS = [1, 16, 16, 16, 16, 16, 1]
    DILATIONS = [1, 2, 4, 8, 16, 32]
elif RECEPTIVE_FIELD == 63:
    CHANNELS = [1, 16, 16, 16, 16, 1]
    DILATIONS = [1, 2, 4, 8, 16]

# ----------------------------- prepare data -----------------------------
x_ = np.linspace(0, 1, RESOLUTION)
y_ = 0.5 * (np.sin(10 * np.pi * x_) + np.sin(23.7 * np.pi * x_))

x = torch.ones(1, 1, RESOLUTION, dtype=torch.float32).to(device) * 0.5
x[0, 0, 0] = 1.0
y = torch.from_numpy(y_).float().view(1, 1, RESOLUTION).to(device)

# -------------------- instantiate model & optimizer ---------------------
resamplings = [nn.ZeroPad1d(((KERNEL_SIZE - 1) * d, 0)) for d in DILATIONS]
activations = [nn.GELU(approximate="tanh")] * (len(CHANNELS) - 2) + [None]

model = DCN(
    CHANNELS,
    activations,
    kernel_size=KERNEL_SIZE,
    stride=1,
    padding=0,
    dilation=DILATIONS,
    resamplings=resamplings,
    dim=1,
)
model.to(device)
init_weights(model, activations[0])
optimizer = torch.optim.AdamW(model.parameters(), LR)

# ------------------------------- training -------------------------------
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

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(train_cost, "k")
plt.show()

fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
ax.plot(x_, y_pred[0, 0].detach().cpu(), "k", linewidth=1.5)
ax.plot(x_, y[0, 0].detach().cpu(), "r--", linewidth=1.5)
fig.tight_layout(pad=0)
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(
    str(BASE_DIR / f"../../results/tcn_sine_{RECEPTIVE_FIELD}.csv"),
    i=np.arange(RESOLUTION) + 1,
    z=x_,
    y=y[0, 0].detach().cpu(),
    y_pred=y_pred[0, 0].detach().cpu(),
)
