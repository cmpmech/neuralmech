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
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 6000
LR = 1e-2
REGULARIZATION = 0
BATCH_SIZE = 3200  # full batch
SAMPLES = 128
NOISE_STD = 0.1


# define loss
def gaussian_nll(y_pred, y_true):
    mean = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]
    var = torch.exp(log_var)
    nll = 0.5 * torch.log(2 * torch.pi * var) + (y_true - mean) ** 2 / (2.0 * var)
    return torch.mean(nll)


cost_fun = gaussian_nll

# model settings
LAYERS = [1, 32, 32, 2]  # second output predicts the input-dependent log-variance
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------ create data ------------------------------------
x_data = torch.rand(SAMPLES) * 2 - 1
y_data = torch.sin(2 * torch.pi * x_data) + torch.randn_like(x_data) * NOISE_STD * (
    x_data + 1.0
)
x_data, y_data = x_data.unsqueeze(1).to(device), y_data.unsqueeze(1).to(device)

dataset = TensorDataset(x_data, y_data)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS).to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
print_every = 10

tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = standardizex(x.to(device)), standardizey(y.to(device))
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
            x, y = standardizex(x.to(device)), standardizey(y.to(device))
            y_pred = model(x)
            val_cost[epoch] += cost_fun(y_pred, y).item()
        val_cost[epoch] /= len(val_loader)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
f = lambda x: torch.sin(2 * torch.pi * x)
x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1).to(device)
y_test = f(x_test)

model.eval()
with torch.no_grad():
    y_pred_test = model(standardizex(x_test))
    mean_test = standardizey.inverse(y_pred_test[:, 0:1]).cpu().numpy()
    var_test = torch.exp(y_pred_test[:, 1:2]).cpu().numpy() * (
        standardizey.x_std.item() ** 2
    )
    std_test = np.sqrt(var_test)

x_train_arr = train_data.dataset.tensors[0][train_data.indices].squeeze().cpu().numpy()
y_train_arr = train_data.dataset.tensors[1][train_data.indices].squeeze().cpu().numpy()

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test.cpu(), y_test.cpu(), "k")
    ax.fill_between(
        x_test.squeeze().cpu(),
        (mean_test - 2 * std_test).squeeze(),
        (mean_test + 2 * std_test).squeeze(),
        alpha=0.3,
        color="r",
    )
    ax.plot(x_train_arr, y_train_arr, "ko")
    ax.plot(x_test.cpu(), mean_test, "r--")
    ax.set_ylim(-2, 2)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "mle_hetero.csv",
        x=x_test.squeeze().cpu().numpy(),
        y=y_test.squeeze().cpu().numpy(),
        mean=mean_test.squeeze(),
        std=std_test.squeeze(),
    )
    save_csv(RESULTS_DIR / "mle_hetero_train.csv", x=x_train_arr, y=y_train_arr)
