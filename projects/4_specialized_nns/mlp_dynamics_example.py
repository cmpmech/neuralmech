import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, differentiate, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# --------------------------------- training settings ---------------------------------
LR = 2e-3
EPOCHS = 2000  # 20
REGULARIZATION = 0
BATCH_SIZE = 10

SAMPLES_TRAIN, TMAX_TRAIN = 10, 1.5
SAMPLES_VAL, TMAX_VAL = 100, 18


# define loss
cost_fun = nn.MSELoss(reduction="mean")

# ----------------------------------- model settings ----------------------------------
layers = [2, 48, 48, 2]
activations = [nn.GELU(approximate="tanh")] * (len(layers) - 2)

# ---------------------------------- data generation ----------------------------------
k, m, du0dt = 10, 1, 1

omega = np.sqrt(k / m)
A = du0dt / omega
phi = 0  # u0 = 0

u_fun = lambda t: A * torch.sin(omega * t + phi)
dudt_fun = lambda t: omega * A * torch.cos(omega * t + phi)
ddudtt_fun = lambda t: -(omega**2) * A * torch.sin(omega * t + phi)


def create_dataset(tmax, samples):
    t = torch.linspace(0, tmax, samples, dtype=torch.float32)
    u = u_fun(t).unsqueeze(1)
    p = m * dudt_fun(t).unsqueeze(1)
    dpdt = m * ddudtt_fun(t).unsqueeze(1)
    dataset = TensorDataset(torch.hstack((u, p)), torch.hstack((p / m, dpdt)))
    return dataset


train_data = create_dataset(TMAX_TRAIN, SAMPLES_TRAIN)
val_data = create_dataset(TMAX_VAL, SAMPLES_VAL)

# ------------------------ instantiate model & prepare training -----------------------
model = MLP(layers, activations)
init_weights(model, activations[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch

    model.eval()
    for x, y in val_loader:
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        val_cost[epoch] += cost.item()
    val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")


# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
plt.show()

# trajectory prediction
dt = 0.02
t = np.arange(0, TMAX_VAL, dt)
N = len(t)


def system(t, y):
    x = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # shape [1, 2]
    with torch.no_grad():
        dydt = model(x)
    return dydt[0].tolist()  # [du/dt, dp/dt]


sol = solve_ivp(
    system,
    (0, TMAX_VAL),
    [0, m * du0dt],  # initial conditions
    t_eval=t,
    method="Radau",
)
u, p = sol.y


t_train = np.linspace(0, TMAX_TRAIN, SAMPLES_TRAIN)
u_train = train_data.tensors[0][:, 0]
p_train = train_data.tensors[0][:, 1]

energy_kin = 0.5 / m * p**2
energy_pot = 0.5 * k * u**2
energy = energy_kin + energy_pot

fig, ax = plt.subplots(1, 3)
ax[0].plot(t, u, "k")
ax[0].plot(t, p, "r")
ax[0].plot(t_train, u_train, "ko")
ax[0].plot(t_train, p_train, "ro")
ax[1].plot(u, p, "k")
ax[2].plot(t, energy, "k")
ax[2].plot(t, energy_kin, "r")
ax[2].plot(t, energy_pot, "b")
plt.show()


save_csv(
    RESULTS_DIR / f"mlp_dynamics_{EPOCHS}.csv",
    t=t,
    u=u,
    p=p,
    e=energy,
    ek=energy_kin,
    ep=energy_pot,
    utrue=u_fun(torch.from_numpy(t)),
    dudttrue=dudt_fun(torch.from_numpy(t)),
)

save_csv(
    RESULTS_DIR / f"mlp_dynamics_train.csv",
    t=t_train,
    u=u_train,
    p=p_train,
)
