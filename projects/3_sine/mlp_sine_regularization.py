import argparse
import copy
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, flatten_params, get_params, init_weights
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
EPOCHS = 1000  # 400
LR = 1e-2
REGULARIZATION = 0  # 0 # 1e0  # 5e0  # 1e1
DROPOUT = 0.0  # 0.1
BATCH_SIZE = 32

PATIENCE = 200  # None

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
# three hidden layers is sufficient (five to show overfitting)
LAYERS = [1, 24, 24, 24, 24, 1]
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
dropouts = [DROPOUT] * (len(LAYERS) - 2) if DROPOUT else None
model = MLP(LAYERS, ACTIVATIONS, dropouts=dropouts)
model.to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
weight_rms = [0] * EPOCHS
tic = time.time()
print_every = 10
best_val = float("inf")
epochs_since_improve = 0
best_state = None
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    weights = flatten_params(get_params(model, kind="weights"))
    weight_rms[epoch] = torch.norm(weights).item() / math.sqrt(len(weights))

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

    if PATIENCE is not None:
        if val_cost[epoch] < best_val:
            best_val = val_cost[epoch]
            epochs_since_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"early stopping at epoch {epoch} (best val {best_val:.2e})")
                train_cost = train_cost[: epoch + 1]
                val_cost = val_cost[: epoch + 1]
                weight_rms = weight_rms[: epoch + 1]
                break

if PATIENCE is not None and best_state is not None:
    last_state = copy.deepcopy(model.state_dict())  # normally not needed
    model.load_state_dict(best_state)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1)
y_test = torch.sin(2 * torch.pi * x_test)
model.eval()
with torch.no_grad():
    y_best_test = model(standardizex(x_test).to(device))
    y_best_test = standardizey.inverse(y_best_test).cpu().numpy()
    if PATIENCE is not None:
        model.load_state_dict(last_state)
        y_last_test = model(standardizex(x_test).to(device))
        y_last_test = standardizey.inverse(y_last_test).cpu().numpy()
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, "k")
    ax.plot(X_train, Y_train, "ko")
    ax.plot(X_val, Y_val, "ro")
    ax.plot(x_test, y_best_test, "r--")
    if PATIENCE is not None:
        ax.plot(x_test, y_last_test, "b:")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    ax2 = ax.twinx()
    ax2.plot(weight_rms, "b")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR
        / f"mlp_sine_regularization_{PATIENCE}_{REGULARIZATION}_{DROPOUT}.csv",
        ct=train_cost,
        cv=val_cost,
        w=weight_rms,
    )

    if PATIENCE is None and DROPOUT == 0.0:
        save_csv(
            RESULTS_DIR / f"mlp_sine_regularization_pred_{REGULARIZATION}.csv",
            x=x_test.flatten(),
            y=y_best_test.flatten(),
        )
