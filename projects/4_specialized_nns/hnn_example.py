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
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
LR = 2e-3
EPOCHS = 2000  # 20
REGULARIZATION = 0
BATCH_SIZE = 10

SAMPLES_TRAIN, TMAX_TRAIN = 10, 1.5
SAMPLES_VAL, TMAX_VAL = 100, 18


# define loss
def cost_fun(H_pred, u, p, dpdt, m):
    dudt = p / m
    cost = torch.mean(
        (differentiate(H_pred, p) - dudt) ** 2 + (differentiate(H_pred, u) + dpdt) ** 2
    )
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
    p = m * dudt_fun(t).unsqueeze(1)
    dpdt = m * ddudtt_fun(t).unsqueeze(1)
    dataset = TensorDataset(u, p, dpdt)
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
    for u, p, dpdt in train_loader:
        optimizer.zero_grad()
        u.requires_grad, p.requires_grad = True, True  # to enable differentiation
        H_pred = model(torch.hstack((u, p)))
        cost = cost_fun(H_pred, u, p, dpdt, m)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch

    model.eval()
    for u, p, dpdt in val_loader:
        u.requires_grad, p.requires_grad = True, True  # to enable differentiation
        H_pred = model(torch.hstack((u, p)))
        cost = cost_fun(H_pred, u, p, dpdt, m)
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
    pi = torch.tensor([y[1]], requires_grad=True, dtype=torch.float32)
    H = model(torch.hstack((ui, pi)))
    dHdu = differentiate(H, ui, graph=True).item()
    dHdp = differentiate(H, pi, graph=False).item()
    return [dHdp, -dHdu]


sol = solve_ivp(
    system,
    (0, TMAX_VAL),
    [0, m * du0dt],  # initial conditions
    t_eval=t,
    method="Radau",
)
u, p = sol.y


t_train = np.linspace(0, TMAX_TRAIN, SAMPLES_TRAIN)
u_train = train_data.tensors[0][:, 0].numpy()
p_train = train_data.tensors[1][:, 0].numpy()

energy_kin = 0.5 / m * p**2
energy_pot = 0.5 * k * u**2
energy = energy_kin + energy_pot


H = model(
    torch.from_numpy(np.hstack((np.expand_dims(u, 1), np.expand_dims(p, 1)))).to(
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
    ax[0].plot(t, p, "r")
    ax[0].plot(t_train, u_train, "ko")
    ax[0].plot(t_train, p_train, "ro")
    ax[1].plot(u, p, "k")
    ax[2].plot(t, energy, "k")
    ax[2].plot(t, energy_kin, "r")
    ax[2].plot(t, energy_pot, "b")
    ax[2].plot(t, H, "k--")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / f"hnn_{EPOCHS}.csv",
        t=t,
        u=u,
        p=p,
        H=H.numpy(),
        e=energy,
        ek=energy_kin,
        ep=energy_pot,
        utrue=u_fun(torch.from_numpy(t)),
        dudttrue=dudt_fun(torch.from_numpy(t)),
    )
    save_csv(
        CSV_DIR / "hnn_train.csv",
        t=t_train,
        u=u_train,
        p=p_train,
    )
