import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 1000  # full batch, per task
LR = 1e-2

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, 24, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# forgetting
SAMPLES = 400

f = lambda x: np.sin(2 * np.pi * x)

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "sine.npz")
X_train = torch.from_numpy(data["X"]).to(torch.float32)
Y_train = torch.from_numpy(data["Y"]).to(torch.float32)
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

left = X_train[:, 0] < 0
tasks = {"left": (X_train[left], Y_train[left]), "right": (X_train[~left], Y_train[~left])}
x_test = torch.linspace(-1, 1, SAMPLES)[:, None]

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS).to(device)
init_weights(model, ACTIVATIONS[0])

# ------------------------------------- training --------------------------------------
y_preds = {}
tic = time.time()
for task, (X, Y) in tasks.items():
    optimizer = torch.optim.Adam(model.parameters(), LR)
    x, y = standardizex(X.to(device)), standardizey(Y.to(device))
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        optimizer.zero_grad()
        cost = cost_fun(model(x), y)
        cost.backward()
        optimizer.step()
        pbar.set_postfix({"train": f"{cost.item():.2e}"})

    with torch.no_grad():
        y_pred = standardizey.inverse(model(standardizex(x_test.to(device))))
    y_preds[task] = y_pred.squeeze().cpu().numpy()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = x_test.squeeze().numpy()
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
for ax, (task, (X, Y)) in zip(axs, tasks.items()):
    ax.plot(x_test, f(x_test), "k--")
    ax.plot(x_test, y_preds[task], "r")
    ax.plot(X, Y, "ko", markersize=3)
    ax.set_ylim(-2, 2)
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "catastrophic_forgetting.csv", x=x_test, y=f(x_test),
             y_pred_left=y_preds["left"], y_pred_right=y_preds["right"])
    for task, (X, Y) in tasks.items():
        save_csv(CSV_DIR / f"catastrophic_forgetting_{task}.csv", x=X[:, 0], y=Y[:, 0])
plt.close()
