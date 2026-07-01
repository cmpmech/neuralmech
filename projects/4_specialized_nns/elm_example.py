import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from DL import Standardizer, init_weights
from NN import MLP, ELM
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
REGULARIZATION = 0  # ridge penalty on the closed-form readout
USE_TRAINING = False  # also gradient-train the readout after the closed-form fit

EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 32

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, 24, 24, 24]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 1)]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.float32),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

# --------------------------- instantiate model & optimizer ---------------------------
backbone = MLP(LAYERS, post_modules=ACTIVATIONS)
backbone.to(device)
init_weights(backbone, ACTIVATIONS[0])

model = ELM(backbone, LAYERS[-1], 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------------------- fit ----------------------------------------
tic = time.time()
model.fit(standardizex(X_train), standardizey(Y_train), REGULARIZATION)
toc = time.time()
print(f"elapsed fitting time {toc - tic:.2e} s")

if USE_TRAINING:
    train_cost = [0] * EPOCHS
    for epoch in range(EPOCHS):
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
        print(f"{train_cost[epoch]:.2e}")

# ----------------------------------- postprocessing ----------------------------------
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]

model.eval()
with torch.no_grad():
    y_val_pred = standardizey.inverse(model(standardizex(X_val)))
    cost = cost_fun(y_val_pred, Y_val)
print(f"validation cost {cost:.2e}")

f = lambda x: torch.sin(2 * torch.pi * x)
x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1)
y_test = f(x_test)

with torch.no_grad():
    y_pred = standardizey.inverse(model(standardizex(x_test)))

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, "k")
    ax.plot(x_test, y_pred.detach().cpu(), "r--")
    ax.plot(X_train, Y_train, "bo")
    ax.set_ylim(-2, 2)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "elm_sine_test.csv",
        x=x_test[:, 0],
        y=y_test[:, 0],
        ypred=y_pred[:, 0],
    )
    save_csv(
        CSV_DIR / "elm_sine_train.csv",
        x=X_train[:, 0],
        y=Y_train[:, 0],
    )
