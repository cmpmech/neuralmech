import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
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
EPOCHS = 1000  # full batch
LR = 1e-3

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [2, 64, 64, 2]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]

# physics
DT = 0.1  # time step learned by the network
SAMPLES = 2000
STATE_RANGE = 3.0  # training states cover [-3, 3] in q and p
T = 3000  # rollout steps
Q0 = 2.6  # initial angle, released from rest


# -------------------------------------- helper ---------------------------------------
def rhs(state):
    q, p = state[..., 0], state[..., 1]
    return torch.stack([p, -torch.sin(q)], dim=-1)


def step(state, substeps=20):  # exact reference, fourth-order Runge-Kutta
    h = DT / substeps
    for _ in range(substeps):
        k1 = rhs(state)
        k2 = rhs(state + h / 2 * k1)
        k3 = rhs(state + h / 2 * k2)
        k4 = rhs(state + h * k3)
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def energy(state):
    return 0.5 * state[..., 1] ** 2 + 1 - torch.cos(state[..., 0])


# ------------------------------------ create data ------------------------------------
X_train = STATE_RANGE * (2 * torch.rand(SAMPLES, 2, dtype=torch.float64) - 1)
Y_train = step(X_train)
X_train, Y_train = X_train.float(), Y_train.float()

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS).to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# ------------------------------------- training --------------------------------------
x, y = X_train.to(device), Y_train.to(device)
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    optimizer.zero_grad()
    cost = cost_fun(x + DT * model(x), y)  # the network predicts the rate of change
    cost.backward()
    optimizer.step()
    scheduler.step()
    pbar.set_postfix({"train": f"{cost.item():.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------------- rollout ---------------------------------------
reference = [torch.tensor([Q0, 0.0], dtype=torch.float64)]
prediction = [torch.tensor([Q0, 0.0])]
with torch.no_grad():
    for tau in range(T):
        reference.append(step(reference[-1]))
        state = prediction[-1].to(device)
        prediction.append((state + DT * model(state)).cpu())
reference = torch.stack(reference).float()
prediction = torch.stack(prediction)

single_step = (prediction[1] - reference[1]).norm() / (reference[1] - reference[0]).norm()
print(f"relative error of the first step {single_step:.2e}")
print(f"energy reference {energy(reference[0]):.3f}, prediction after {T} steps {energy(prediction[-1]):.3f}")

# ----------------------------------- postprocessing ----------------------------------
t = DT * np.arange(T + 1)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(prediction[:, 0], prediction[:, 1], "r", linewidth=0.5)
axs[0].plot(reference[:, 0], reference[:, 1], "k--")
axs[0].set_aspect("equal")
axs[1].plot(t, energy(prediction), "r")
axs[1].plot(t, energy(reference), "k--")
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.25)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "rollout_drift.csv", t=t, q=reference[:, 0], p=reference[:, 1],
             e=energy(reference), q_pred=prediction[:, 0], p_pred=prediction[:, 1],
             e_pred=energy(prediction))
plt.close()
