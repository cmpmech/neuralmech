import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import MLP, SIRENsine
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
USE_SIREN = True
# USE_SIREN = False

EPOCHS = 2000
LR = 1e-2  # 5e-3
BATCH_SIZE = 32
FREQ = 3
SAMPLES = 256

SHARPNESS = 4.0
# SHARPNESS = 8.0
# SHARPNESS = 16.0

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, 32, 32, 1]
OMEGA_0 = 2.0
# OMEGA_0 = 10.0
# OMEGA_0 = 15.0

# ------------------------------------ prepare data -----------------------------------
X = torch.linspace(-1, 1, SAMPLES).unsqueeze(1)
Y = torch.tanh(SHARPNESS * torch.sin(2 * torch.pi * FREQ * X))

dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

X_train = dataset.tensors[0]
Y_train = dataset.tensors[1]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)


# --------------------------- instantiate model & optimizer ---------------------------
if USE_SIREN:
    activation = SIRENsine(omega_0=OMEGA_0)
else:
    activation = nn.ReLU(inplace=True)
    # activation = nn.GELU(approximate="tanh")

activations = [activation for _ in range(len(LAYERS) - 2)] + [None]
model = MLP(LAYERS, post_modules=activations)
model.to(device)
init_weights(model, activation)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -------------------------------------- training -------------------------------------
train_cost = [0.0] * EPOCHS
tic = time.time()
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x, y = standardizex(x), standardizey(y)
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    scheduler.step()
    if epoch % 50 == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time: {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
f = lambda x: torch.tanh(SHARPNESS * torch.sin(2 * torch.pi * FREQ * x))

x_test = torch.linspace(-1, 1, 2 * SAMPLES).unsqueeze(1)
y_test = f(x_test)

with torch.no_grad():
    y_pred = standardizey.inverse(model(standardizex(x_test)))
    y_pred_train = standardizey.inverse(model(standardizex(X_train)))

x_grad = x_test.detach().requires_grad_(True)
dy_pred = grad(
    standardizey.inverse(model(standardizex(x_grad))),
    x_grad,
    torch.ones_like(x_grad),
)[0]

dy_test = (
    (1 - torch.tanh(SHARPNESS * torch.sin(2 * torch.pi * FREQ * x_test)) ** 2)
    * SHARPNESS
    * 2
    * torch.pi
    * FREQ
    * torch.cos(2 * torch.pi * FREQ * x_test)
)

tag = f"siren_{SHARPNESS}_{USE_SIREN}_{OMEGA_0}"

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, "k")
    ax.plot(x_test, y_pred.detach().cpu(), "r--")
    ax.plot(X_train, Y_train, "bo")
    ax.set_ylim(-2, 2)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test, dy_test, "k")
    ax.plot(x_grad.detach(), dy_pred.detach(), "r--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / f"{tag}_test.csv",
        x=x_test[:, 0],
        y=y_test[:, 0],
        ypred=y_pred[:, 0],
    )
    save_csv(
        CSV_DIR / f"{tag}_grad.csv",
        x=x_grad.detach()[:, 0],
        dy=dy_test[:, 0],
        dypred=dy_pred.detach()[:, 0],
    )
    save_csv(
        CSV_DIR / f"{tag}_train.csv",
        x=X_train[:, 0],
        y=Y_train[:, 0],
        ypred=y_pred_train[:, 0],
    )
