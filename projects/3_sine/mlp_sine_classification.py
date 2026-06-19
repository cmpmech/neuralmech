import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, init_weights
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
EPOCHS = 200
LR = 1e-2
REGULARIZATION = 1e-2
BATCH_SIZE = 32

# define loss
cost_fun = nn.CrossEntropyLoss(reduction="mean")

# model settigns
CLASSES = 3  # has to fit to the data generation
LAYERS = [
    1,
    24,
    24,
    24,
    CLASSES,
]  # three hidden layers is sufficient (five to show overfitting)
ACTIVATIONS = [torch.nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "discrete_sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.long),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

test_data = np.load(DATA_DIR / "discrete_sine_test.npz")
x_test = torch.from_numpy(test_data["X"]).to(torch.float32).to(device)
y_test = torch.from_numpy(test_data["Y"]).to(torch.long).to(device)

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=0)

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
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = standardizex(x)
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
            x = standardizex(x)
            y_pred = model(x)
            cost = cost_fun(y_pred, y)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"Elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
model.eval()
with torch.no_grad():
    y_pred_test = torch.argmax(model(standardizex(x_test)), dim=-1)

Y_train = train_data.dataset.tensors[1][train_data.indices]
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test.cpu(), y_test.cpu(), "k")
    ax.plot(X_train, Y_train, "ko")
    ax.plot(X_val, Y_val, "ro")
    ax.plot(x_test.cpu(), y_pred_test.cpu(), "b.")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "mlp_discrete_sine_test.csv",
        x=x_test[:, 0],
        y=y_test,
        ypred=y_pred_test,
    )
    save_csv(RESULTS_DIR / "mlp_discrete_sine_train.csv", x=X_train[:, 0], y=Y_train)
    save_csv(RESULTS_DIR / "mlp_discrete_sine_val.csv", x=X_val[:, 0], y=Y_val)
