import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, differentiate, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 400
LR = 1e-2
REGULARIZATION = 0
BATCH_SIZE = 32

# define loss
mse = nn.MSELoss(reduction="sum")


def cost_fun(y_pred, dy_pred, y, dy):
    return (mse(y_pred, y) + mse(dy_pred, dy)) / (len(y) + len(dy))


# model settings
# three hidden layers is sufficient (five to show overfitting)
LAYERS = [1, 24, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "sobolev_sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.float32),
    torch.from_numpy(data["DY"]).to(torch.float32),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
DY_train = train_data.dataset.tensors[2][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)
# standardize dy as y

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for x, y, dy in train_loader:
        optimizer.zero_grad()
        x, y, dy = x.to(device), y.to(device), dy.to(device)
        x.requires_grad = True  # to enable differentiation
        y, dy = standardizey(y), standardizey(dy)
        y_pred = model(standardizex(x))
        dy_pred = differentiate(y_pred, x)
        cost = cost_fun(y_pred, dy_pred, y, dy)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch

    model.eval()
    for x, y, dy in val_loader:
        x, y, dy = x.to(device), y.to(device), dy.to(device)
        x.requires_grad = True  # to enable differentiation
        y_pred = model(standardizex(x))
        y, dy = standardizey(y), standardizey(dy)
        dy_pred = differentiate(y_pred, x, graph=False)
        cost = cost_fun(y_pred, dy_pred, y, dy)
        val_cost[epoch] += cost.item()
    val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1)
y_test = torch.sin(2 * torch.pi * x_test)
model.eval()
with torch.no_grad():
    y_pred_test = model(standardizex(x_test).to(device))
    y_pred_test = standardizey.inverse(y_pred_test).cpu().numpy()
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, "k")
    ax.plot(X_train, Y_train, "ko")
    ax.plot(X_val, Y_val, "ro")
    ax.plot(x_test, y_pred_test, "r--")
    plt.show()

# ------------------------------- gradient postprocessing -----------------------------
x_test.requires_grad = True
y_pred = model(standardizex(x_test).to(device))
y_pred = standardizey.inverse(y_pred)

dy_pred = grad(
    y_pred, x_test, torch.ones_like(x_test), retain_graph=True, create_graph=True
)[0]
ddy_pred = grad(
    dy_pred, x_test, torch.ones_like(x_test), retain_graph=True, create_graph=True
)[0]
dddy_pred = grad(
    ddy_pred, x_test, torch.ones_like(x_test), retain_graph=True, create_graph=True
)[0]

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test.detach(), y_pred.detach(), "k")
    ax.plot(x_test.detach(), dy_pred.detach() / 2 / np.pi, "r")
    ax.plot(x_test.detach(), ddy_pred.detach() / 4 / np.pi**2, "b")
    ax.plot(x_test.detach(), dddy_pred.detach() / 8 / np.pi**3, "g")
    plt.show()

# ---------------------------- gradient book postprocessing ---------------------------
else:
    if EPOCHS == 400:
        save_csv(
            RESULTS_DIR / "mlp_sine_sobolev_grad.csv",
            x=x_test.detach()[:, 0],
            y=y_pred.detach()[:, 0],
            dy=dy_pred.detach()[:, 0] / 2 / np.pi,
            ddy=ddy_pred.detach()[:, 0] / 4 / np.pi**2,
            dddy=dddy_pred.detach()[:, 0] / 8 / np.pi**3,
        )

    save_csv(
        RESULTS_DIR / "mlp_sine_sobolev_train.csv",
        x=X_train[:, 0],
        y=Y_train[:, 0],
        dy=DY_train[:, 0] / 2 / np.pi,
    )
