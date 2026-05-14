import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from NN import ICNN
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

device = torch.device("cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# --------------------------- hyperparameters ----------------------------
EPOCHS = 3000
LR = 1e-2
N_TRAIN = 16
NOISE = 0.3
LAYERS = [1, 16, 16, 16, 1]
USE_NONNEG = True  # False

# all must be convex and non-decreasing for ICNN output to be convex in x
activation = torch.nn.Softplus()
# activation = torch.nn.ReLU()
# activation = torch.nn.ELU()
# activation = torch.nn.LeakyReLU(negative_slope=0.1)

cost_fun = nn.MSELoss()

# ------------------------------- data -----------------------------------
x_train = torch.linspace(-2, 2, N_TRAIN).unsqueeze(1).to(device)
y_train = (x_train**2 + NOISE * torch.randn_like(x_train)).to(device)

# -------------------------- model + optimizer ---------------------------
activations = [activation] * (len(LAYERS) - 2)
model = ICNN(LAYERS, activations).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------------------ training --------------------------------
train_cost = [0.0] * EPOCHS
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x_train)
    cost = cost_fun(y_pred, y_train)
    cost.backward()
    optimizer.step()
    if USE_NONNEG:
        model.clamp_z_()
    train_cost[epoch] = cost.item()
    if epoch % 100 == 0:
        pbar.set_postfix({"train": f"{cost.item():.2e}"})

# --------------------------- postprocessing -----------------------------
x_test = torch.linspace(-2.5, 2.5, 200).unsqueeze(1).to(device)
y_test = x_test**2
with torch.no_grad():
    y_test_pred = model(x_test)

fig, ax = plt.subplots()
ax.plot(x_test.cpu(), y_test.cpu(), "k")
ax.plot(x_test.cpu(), y_test_pred.detach().cpu(), "r--")
ax.plot(x_train.cpu(), y_train.cpu(), "bo")

if args.book:
    save_csv(
        RESULTS_DIR / f"icnn_parabola_test_{USE_NONNEG}.csv",
        x=x_test[:, 0].cpu(),
        y=y_test[:, 0].cpu(),
        ypred=y_test_pred[:, 0].cpu(),
    )
    save_csv(
        RESULTS_DIR / "icnn_parabola_train.csv",
        x=x_train[:, 0].cpu(),
        y=y_train[:, 0].cpu(),
    )
    plt.close()
else:
    plt.show()
