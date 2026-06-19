import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import count_kan_params, get_kan_edge_activations
from NN import KAN
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 256

EPOCHS = 1000
LR = 2e-2

# define loss
cost_fun = nn.MSELoss()

# model settings
LAYERS = [1, 6, 6, 1]
SPLINE_ORDER, GRID_SIZE = 3, 5  # default is 3, 5
BASE_ACTIVATION = nn.SiLU

# ------------------------------------ prepare data -----------------------------------
x_ = np.linspace(-1, 1, RESOLUTION)
y = np.sin(8 * np.pi * x_) * np.sin(6 * np.pi * x_)
y = torch.from_numpy(y).to(torch.float32).unsqueeze(1).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = KAN(
    LAYERS,
    spline_order=SPLINE_ORDER,
    grid_size=GRID_SIZE,
    base_activation=BASE_ACTIVATION,
)
model.to(device)
x = torch.from_numpy(x_).to(torch.float32).unsqueeze(1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x)
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
base_params, spline_params = count_kan_params(model)
total_params = base_params + spline_params
print(f"linear {base_params:d} & spline {spline_params:d}: total {total_params:d}")

acts_x_0, acts_0 = get_kan_edge_activations(model, 0)
acts_x_1, acts_1 = get_kan_edge_activations(model, 1)
acts_x_2, acts_2 = get_kan_edge_activations(model, 2)

# edge strengths layer 1: L1 norm of full edge function (pruning metric)
edge_strength_1 = acts_1.abs().mean(-1)  # (out, in)
np.set_printoptions(precision=3, suppress=True)
print("edge strength layer 1 (out x in):\n", edge_strength_1.numpy())

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

    fig, ax = plt.subplots(dpi=200)
    ax.plot(x_, y.cpu(), "k", linewidth=1.5)
    ax.plot(x_, y_pred.detach().cpu(), "r--", linewidth=1.5)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "kan_fit.csv",
        x=x_,
        y=y.squeeze().cpu(),
        ypred=y_pred.squeeze().detach().cpu(),
    )
    for layer_idx, (acts_x, acts) in enumerate(
        [(acts_x_0, acts_0), (acts_x_1, acts_1), (acts_x_2, acts_2)]
    ):
        cols = {f"x{j}": acts_x[0, j] for j in range(acts_x.shape[1])}
        cols.update(
            {
                f"phi{i}{j}": acts[i, j]
                for i in range(acts.shape[0])
                for j in range(acts.shape[1])
            }
        )
        save_csv(RESULTS_DIR / f"kan_edges_{layer_idx}.csv", **cols)
        strength = acts.abs().mean(-1)  # (out, in)
        strength = 0.1 + 0.9 * (strength - strength.min()) / (
            strength.max() - strength.min()
        )
        save_csv(
            RESULTS_DIR / f"kan_strength_{layer_idx}.csv",
            **{f"phi{j}": strength[:, j] for j in range(strength.shape[1])},
        )
