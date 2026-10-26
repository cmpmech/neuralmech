import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import init_weights
from NN import MLP

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
DEVICES = ["cuda:0", "cuda:1"]  # ["cuda:0", "cuda:0"] splits on one GPU

# hyperparameters
EPOCHS = 100
LR = 1e-2
BATCH_SIZE = 64

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, 24, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# data
SAMPLES = 1024
NOISE = 0.2

# ------------------------------------ create data ------------------------------------
X = 2 * torch.rand(SAMPLES, 1) - 1
Y = torch.sin(2 * torch.pi * X) + NOISE * (2 * torch.rand(SAMPLES, 1) - 1)
train_loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS)
first = model.model[:4].to(DEVICES[0])  # first two layers
second = model.model[4:].to(DEVICES[1])  # last two layers
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    train_cost = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = second(first(x.to(DEVICES[0])).to(DEVICES[1]))
        cost = cost_fun(y_pred, y.to(DEVICES[1]))
        cost.backward()
        optimizer.step()
        train_cost += cost.item()
    pbar.set_postfix({"train": f"{train_cost / len(train_loader):.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
model.eval()
with torch.no_grad():
    y_pred_test = second(first(x_test.to(DEVICES[0])).to(DEVICES[1])).cpu()

fig, ax = plt.subplots()
ax.plot(x_test, torch.sin(2 * torch.pi * x_test), "k")
ax.plot(X, Y, "ko", markersize=2)
ax.plot(x_test, y_pred_test, "r--")
plt.show()
