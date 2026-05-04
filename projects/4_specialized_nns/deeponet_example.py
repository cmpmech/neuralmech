import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer
from NN import MLP, DeepONet
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------- training settings ---------------------------
# query resolution can change between train and test;
# sensor count is fixed by the branch architecture
TRAIN_RES = 32
TEST_RES = 64  # 32 64 256
SENSORS = 32

EPOCHS = 200
LR = 2e-2
BATCH_SIZE = 24

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
P = 16  # latent basis dimension
branch_layers = [SENSORS, 16, 16, P]
trunk_layers = [1, 16, 16, P]
branch_acts = [nn.GELU(approximate="tanh")] * (len(branch_layers) - 2)
trunk_acts = [nn.GELU(approximate="tanh")] * (len(trunk_layers) - 2)

# ----------------------------- prepare data -----------------------------
data = np.load(DATA_DIR / f"deeponet_sine_{TRAIN_RES}.npz")
G_train = torch.from_numpy(data["G"]).to(torch.float32)  # (N, S) sensor values
X_train = torch.from_numpy(data["X"]).to(torch.float32)  # (N, R) query coords
Y_train = torch.from_numpy(data["Y"]).to(torch.float32)  # (N, R) targets

data_test = np.load(DATA_DIR / f"deeponet_sine_{TEST_RES}.npz")
G_test = torch.from_numpy(data_test["G"]).to(torch.float32)
X_test = torch.from_numpy(data_test["X"]).to(torch.float32)
Y_test = torch.from_numpy(data_test["Y"]).to(torch.float32)

dataset = TensorDataset(G_train, X_train, Y_train)

# standardization (fit on train, applied to test)
standardizeg = Standardizer(G_train, dim=(0, 1))  # global per-channel
standardizey = Standardizer(Y_train, dim=(0, 1))  # global per-channel

# ----------------- instantiate model & prepare training -----------------
branch_model = MLP(branch_layers, branch_acts)
trunk_model = MLP(trunk_layers, trunk_acts)
model = DeepONet(branch_model, trunk_model)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------- training -------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for g, x, y in train_loader:
        g, x, y = g.to(device), x.to(device), y.to(device)
        g, y = standardizeg(g), standardizey(y)

        # replicate g per query so branch and trunk batch sizes match
        B, R = x.shape
        g_flat = g.unsqueeze(1).expand(B, R, -1).reshape(B * R, SENSORS)
        x_flat = x.reshape(B * R, 1)
        y_flat = y.reshape(B * R, 1)

        optimizer.zero_grad()
        y_pred = model(x_flat, g_flat)
        cost = cost_fun(y_pred, y_flat)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")


# ---------------------------- postprocessing ----------------------------
model.eval()


def predict(g_sample, x_sample):
    R = x_sample.shape[0]
    g = standardizeg(g_sample.unsqueeze(0).to(device)).expand(R, SENSORS)
    x = x_sample.to(device).unsqueeze(1)
    with torch.no_grad():
        return standardizey.inverse(model(x, g)).cpu().squeeze()


g_train_s, x_train_s, y_train_s = G_train[0], X_train[0], Y_train[0]
g_test_s, x_test_s, y_test_s = G_test[0], X_test[0], Y_test[0]

y_train_pred = predict(g_train_s, x_train_s)
y_test_pred = predict(g_test_s, x_test_s)

# query points are random; sort for line plots
idx_train = torch.argsort(x_train_s)
idx_test = torch.argsort(x_test_s)

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_train_s[idx_train], y_train_s[idx_train], "k")
    ax.plot(x_train_s[idx_train], y_train_pred[idx_train], "r--")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test_s[idx_test], y_test_s[idx_test], "k")
    ax.plot(x_test_s[idx_test], y_test_pred[idx_test], "r--")
    plt.show()
else:
# ------------------------- book postprocessing --------------------------
    save_csv(
        RESULTS_DIR / f"deeponet_{TEST_RES}.csv",
        x=x_test_s[idx_test],
        y=y_test_s[idx_test],
        ypred=y_test_pred[idx_test],
    )
