import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, init_weights
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
EPOCHS = 1000
LR = 1e-2
REGULARIZATION = 0
BATCH_SIZE = 32
SAMPLES = 96

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
USE_SYMMETRY = True
USE_STRUCTURE = True
LAYERS = [1, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]


# --------------------------------------- helper --------------------------------------
class SymmetricMLP(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # shared among all inputs

    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return self.encoder(x1) * self.encoder(x2)


class UnSymmetricMLP(nn.Module):
    def __init__(self, encoder1, encoder2):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return self.encoder1(x1) * self.encoder2(x2)


# ------------------------------------ prepare data -----------------------------------
x1 = torch.rand(SAMPLES, 1) * 2 - 1
x2 = torch.rand(SAMPLES, 1) * 2 - 1
y = torch.sin(2 * torch.pi * x1) * torch.sin(2 * torch.pi * x2)

dataset = TensorDataset(torch.cat([x1, x2], 1), y)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 1))  # per-axis would break symmetry
standardizey = Standardizer(Y_train, dim=0)

# --------------------------- instantiate model & optimizer ---------------------------
if not USE_STRUCTURE:
    model = MLP([2] + LAYERS[1:], post_modules=ACTIVATIONS)
elif not USE_SYMMETRY:
    model = UnSymmetricMLP(MLP(LAYERS, post_modules=ACTIVATIONS), MLP(LAYERS, post_modules=ACTIVATIONS))
else:
    model = SymmetricMLP(MLP(LAYERS, post_modules=ACTIVATIONS))
model.to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x, y = standardizex(x), standardizey(y)
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x, y = standardizex(x), standardizey(y)
            y_pred = model(x)
            cost = cost_fun(y_pred, y)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
resolution = 300
x1 = torch.linspace(-1, 1, resolution)
x2 = torch.linspace(-1, 1, resolution)
x1, x2 = torch.meshgrid(x1, x2, indexing="ij")
with torch.no_grad():
    model_input = torch.stack([x1.flatten(), x2.flatten()], dim=1)
    y = standardizey.inverse(model(standardizex(model_input)))
y = y.reshape(resolution, resolution)

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "r")
    ax.plot(val_cost, "b")
    plt.show()

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.pcolormesh(x1, x2, y, cmap="Spectral", vmin=-1, vmax=1)
    ax.plot(X_train[:, 0], X_train[:, 1], "ko", markersize=4, alpha=0.4)
    ax.set_aspect("equal")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:

    def save_field(field, path):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        ax.pcolormesh(x1, x2, field, cmap="Spectral", vmin=-1, vmax=1)
        ax.plot(X_train[:, 0], X_train[:, 1], "ko", markersize=4, alpha=0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path)
        plt.close()

    save_field(y, RESULTS_DIR / f"paramsharing_{USE_SYMMETRY}_{USE_STRUCTURE}.png")
    save_field(y.T, RESULTS_DIR / f"paramsharingT_{USE_SYMMETRY}_{USE_STRUCTURE}.png")
