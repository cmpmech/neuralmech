import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
EPOCHS = 4000
LR = 1e-2
LR_MUON = 1e-2
BATCH_SIZE = 32

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
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
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)


# -------------------------------------- training -------------------------------------
def train(use_muon):
    torch.manual_seed(1)
    model = MLP(LAYERS, post_modules=ACTIVATIONS)
    model.to(device)
    init_weights(model, ACTIVATIONS[0])

    if use_muon:
        weights = [p for p in model.parameters() if p.ndim == 2]
        hidden = weights[1:-1]
        rest = [p for p in model.parameters() if all(p is not h for h in hidden)]
        optimizers = [
            torch.optim.Muon(hidden, lr=LR_MUON, weight_decay=0),
            torch.optim.AdamW(rest, lr=LR, weight_decay=0),
        ]
    else:
        optimizers = [torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)]

    train_cost = np.zeros(EPOCHS)
    val_cost = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = standardizex(x), standardizey(y)
            for optimizer in optimizers:
                optimizer.zero_grad()
            cost = cost_fun(model(x), y)
            cost.backward()
            for optimizer in optimizers:
                optimizer.step()
            train_cost[epoch] += cost.item()
        train_cost[epoch] /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x, y = standardizex(x), standardizey(y)
                val_cost[epoch] += cost_fun(model(x), y).item()
            val_cost[epoch] /= len(val_loader)
    return train_cost / train_cost[0], val_cost / val_cost[0]


tic = time.time()
adam_train, adam_val = train(use_muon=False)
muon_train, muon_val = train(use_muon=True)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"final training cost adam {adam_train[-1]:.2e}, muon {muon_train[-1]:.2e}")
print(f"final validation cost adam {adam_val[-1]:.2e}, muon {muon_val[-1]:.2e}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(adam_train, "k")
    ax.plot(adam_val, "k--")
    ax.plot(muon_train, "b")
    ax.plot(muon_val, "b--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "mlp_sine_muon.csv",
        adam_train=adam_train,
        adam_val=adam_val,
        muon_train=muon_train,
        muon_val=muon_val,
    )
