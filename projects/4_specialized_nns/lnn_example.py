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

from DL import differentiate, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(1)

# -------------------------------------- settings -------------------------------------
# hyperparameters
LR = 1e-3
EPOCHS = 500  # 2000  # 20
REGULARIZATION = 0
BATCH_SIZE = 10

SAMPLES_TRAIN, TMAX_TRAIN = 10, 1.5
SAMPLES_VAL, TMAX_VAL = 100, 18


# define loss
def acceleration(L_pred, u, dudt):
    dLdu = differentiate(L_pred, u)
    Hessian = differentiate(dLdu, dudt)
    ddL_dudt2 = differentiate(L_pred, dudt, 2)

    ddudtt = 1.0 / ddL_dudt2 * (dLdu - Hessian * dudt)
    return ddudtt


def cost_fun(L_pred, u, dudt, ddudtt):
    ddudtt_pred = acceleration(L_pred, u, dudt)
    cost = torch.mean((ddudtt_pred - ddudtt) ** 2)
    return cost


# model settings
LAYERS = [2, 48, 48, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------ create data ------------------------------------
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
    dudt = dudt_fun(t).unsqueeze(1)
    ddudtt = ddudtt_fun(t).unsqueeze(1)
    dataset = TensorDataset(u, dudt, ddudtt)
    return dataset


train_data = create_dataset(TMAX_TRAIN, SAMPLES_TRAIN)
val_data = create_dataset(TMAX_VAL, SAMPLES_VAL)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS)
init_weights(model, ACTIVATIONS[0])
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
    for u, dudt, ddudtt in train_loader:
        optimizer.zero_grad()
        u.requires_grad, dudt.requires_grad = True, True  # to enable differentiation
        L_pred = model(torch.hstack((u, dudt)))
        cost = cost_fun(L_pred, u, dudt, ddudtt)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch

    model.eval()
    for u, dudt, ddudtt in val_loader:
        u.requires_grad, dudt.requires_grad = True, True  # to enable differentiation
        L_pred = model(torch.hstack((u, dudt)))
        cost = cost_fun(L_pred, u, dudt, ddudtt)
        val_cost[epoch] += cost.item()
    val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")


# ----------------------------------- postprocessing ----------------------------------
# trajectory prediction
dt = 0.02
t = np.arange(0, TMAX_VAL, dt)


def system(t, y):
    ui = torch.tensor([y[0]], requires_grad=True, dtype=torch.float32)
    dudti = torch.tensor([y[1]], requires_grad=True, dtype=torch.float32)
    L = model(torch.hstack((ui, dudti)))
    ddudtti = acceleration(L, ui, dudti).item()
    return [dudti.item(), ddudtti]  # du/dt = dudt, d(dudt)/dt = ddudtt


sol = solve_ivp(
    system,
    (0, TMAX_VAL),
    [0, du0dt],  # initial conditions
    t_eval=t,
    method="Radau",
)
u, dudt = sol.y

t_train = np.linspace(0, TMAX_TRAIN, SAMPLES_TRAIN)
u_train = train_data.tensors[0][:, 0].numpy()
dudt_train = train_data.tensors[1][:, 0].numpy()

energy_kin = 0.5 * m * dudt**2
energy_pot = 0.5 * k * u**2
energy = energy_kin + energy_pot
L = energy_pot - energy_kin

L_pred = model(
    torch.from_numpy(np.hstack((np.expand_dims(u, 1), np.expand_dims(dudt, 1)))).to(
        torch.float32
    )
).detach()[:, 0]

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(t, u, "k")
    ax[0].plot(t, dudt, "r")
    ax[0].plot(t_train, u_train, "ko")
    ax[0].plot(t_train, dudt_train, "ro")
    ax[1].plot(u, dudt, "k")
    ax[2].plot(t, energy, "k")
    ax[2].plot(t, energy_kin, "r")
    ax[2].plot(t, energy_pot, "b")
    ax[2].plot(t, L_pred, "k--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / f"lnn_{EPOCHS}.csv",
        t=t,
        u=u,
        dudt=dudt,
        L=L,
        Lpred=L_pred.numpy(),
        e=energy,
        ek=energy_kin,
        ep=energy_pot,
        utrue=u_fun(torch.from_numpy(t)),
        dudttrue=dudt_fun(torch.from_numpy(t)),
    )
    save_csv(
        CSV_DIR / "lnn_train.csv",
        t=t_train,
        u=u_train,
        dudt=dudt_train,
    )
