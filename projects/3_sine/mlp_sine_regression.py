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

from DL import Standardizer, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 4000  # 400
LR = 1e-2
REGULARIZATION = 0  # 1e0
BATCH_SIZE = 32

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
# three hidden layers is sufficient (five to show overfitting)
LAYERS = [1, 24, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.float32),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS)
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

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / f"mlp_sine_test_{EPOCHS}.csv",
        x=x_test[:, 0],
        y=y_test[:, 0],
        ypred=y_pred_test[:, 0],
    )
    save_csv(CSV_DIR / "mlp_sine_train.csv", x=X_train[:, 0], y=Y_train[:, 0])
    save_csv(CSV_DIR / "mlp_sine_val.csv", x=X_val[:, 0], y=Y_val[:, 0])
    if EPOCHS == 4000:
        save_csv(
            CSV_DIR / "mlp_sine_cost_history.csv",
            train=np.array(train_cost) / train_cost[0],
            val=np.array(val_cost) / val_cost[0],
        )

# ---------------------------- gradient book postprocessing ---------------------------
if args.book:
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

    if EPOCHS == 400:
        save_csv(
            CSV_DIR / "mlp_sine_grad.csv",
            x=x_test.detach()[:, 0],
            y=y_pred.detach()[:, 0],
            dy=dy_pred.detach()[:, 0] / 2 / np.pi,
            ddy=ddy_pred.detach()[:, 0] / 4 / np.pi**2,
            dddy=dddy_pred.detach()[:, 0] / 8 / np.pi**3,
        )
