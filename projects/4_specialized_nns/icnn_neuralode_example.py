import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from NN import ICNN, NODE
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
EPOCHS = 1000
LR = 1e-2
N_TRAIN = 16
NOISE = 0.3
RHS_LAYERS = [2, 16, 16, 1]  # (state + time) -> dy/dt

# all must be convex and non-decreasing for ICNN rhs to be convex in (y, t)
activation = torch.nn.Softplus()
# activation = torch.nn.ReLU()
# activation = torch.nn.ELU()
# activation = torch.nn.LeakyReLU(negative_slope=0.1)

cost_fun = nn.MSELoss()

# ------------------------------- data -----------------------------------
t_train = torch.linspace(-2, 2, N_TRAIN).to(device)
y_train = (
    (t_train**2 + NOISE * torch.randn_like(t_train)).view(N_TRAIN, 1, 1).to(device)
)
y0 = y_train[0]

# -------------------------- model + optimizer ---------------------------
activations = [activation] * (len(RHS_LAYERS) - 2)
rhs_model = ICNN(RHS_LAYERS, activations)
model = NODE(rhs_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------------------ training --------------------------------
train_cost = [0.0] * EPOCHS
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(t_train, y0)
    cost = cost_fun(y_pred, y_train)
    cost.backward()
    optimizer.step()
    rhs_model.clamp_z_()
    train_cost[epoch] = cost.item()
    if epoch % 100 == 0:
        pbar.set_postfix({"train": f"{cost.item():.2e}"})

# --------------------------- postprocessing -----------------------------
t_test = torch.linspace(-2, 2, 200).to(device)
y_test = (t_test**2).view(-1, 1, 1)
with torch.no_grad():
    y_test_pred = model(t_test, y0)

fig, ax = plt.subplots()
ax.plot(t_test.cpu(), y_test[:, 0, 0].cpu(), "k")
ax.plot(t_test.cpu(), y_test_pred[:, 0, 0].detach().cpu(), "r--")
ax.plot(t_train.cpu(), y_train[:, 0, 0].cpu(), "bo")

if args.book:
    save_csv(
        RESULTS_DIR / "icnn_neuralode_parabola_test.csv",
        x=t_test.cpu(),
        y=y_test[:, 0, 0].cpu(),
        ypred=y_test_pred[:, 0, 0].detach().cpu(),
    )
    save_csv(
        RESULTS_DIR / "icnn_neuralode_parabola_train.csv",
        x=t_train.cpu(),
        y=y_train[:, 0, 0].cpu(),
    )
    plt.close()
else:
    plt.show()
