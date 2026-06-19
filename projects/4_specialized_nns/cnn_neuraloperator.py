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
from NN import DCN
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
TRAIN_RES = 32
TEST_RES = 64  # 32 64 256

EPOCHS = 200
LR = 2e-2
BATCH_SIZE = 24

# define loss
cost_fun = nn.MSELoss()

# model settings
CHANNELS = [1, 16, 16, 16, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(CHANNELS) - 2)]
KERNEL_SIZE, STRIDE, PADDING = 3, 1, 1

# ------------------------------------ prepare data -----------------------------------
data = np.load(DATA_DIR / f"fno_sine_{TRAIN_RES}.npz")
grid = data["x"]
dataset = TensorDataset(
    torch.from_numpy(data["X"]).unsqueeze(1).to(torch.float32),
    torch.from_numpy(data["Y"]).unsqueeze(1).to(torch.float32),
)
X_train = dataset.tensors[0]
Y_train = dataset.tensors[1]

data_test = np.load(DATA_DIR / f"fno_sine_{TEST_RES}.npz")
grid_test = data_test["x"]
X_test = torch.from_numpy(data_test["X"]).unsqueeze(1).to(torch.float32)
Y_test = torch.from_numpy(data_test["Y"]).unsqueeze(1).to(torch.float32)

# standardization
standardizex = Standardizer(X_train, dim=(0, 2))  # global (per channel)
standardizey = Standardizer(Y_train, dim=(0, 2))  # global (per channel)

# --------------------------- instantiate model & optimizer ---------------------------
model = DCN(CHANNELS, ACTIVATIONS, KERNEL_SIZE, STRIDE, PADDING, dim=1)
model.to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.AdamW(model.parameters(), LR)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
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

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")


# ----------------------------------- postprocessing ----------------------------------
model.eval()

x_train, y_train = X_train[0:1], Y_train[0:1]
x_test, y_test = X_test[0:1], Y_test[0:1]

with torch.no_grad():
    y_train_pred = standardizey.inverse(model(standardizex(x_train)))
    y_test_pred = standardizey.inverse(model(standardizex(x_test)))

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(grid, y_train.squeeze(), "k")
    ax.plot(grid, y_train_pred.cpu().squeeze(), "r--")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(grid_test, y_test.squeeze(), "k")
    ax.plot(grid_test, y_test_pred.cpu().squeeze(), "r--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / f"cnn_neuraloperator_{TEST_RES}.csv",
        x=grid_test,
        y=y_test.squeeze(),
        ypred=y_test_pred.cpu().squeeze(),
    )
