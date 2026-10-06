import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from DL import differentiate, init_weights
from NN import MLP
from helper import fem, grid, interpolate, pinn_cost, train
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
DATA = "full"  # full or partial
SOLVER = "pinn"  # partial only, pinn (network for u) or fem (finite elements for u)
EPOCHS = 10000
LR = 1e-2
DECAY = 0.9995  # learning rate factor per epoch
SAMPLES = 100  # collocation points or finite elements
SENSORS = 10  # partial only
NOISE = 0.0
DATA_WEIGHT = 100  # partial only, lifts the sensor misfit to the scale of the residual

# physics
u_fun = lambda x: torch.cos(2 * torch.pi * x)
EA_fun = lambda x: (torch.cos(2 * torch.pi * x) + 3) / 4

# model settings
LAYERS = [1, 32, 32, 1]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
x, w = grid([0.0], [1.0], SAMPLES)
x0 = torch.zeros(1, 1, requires_grad=True)
x1 = torch.ones(1, 1, requires_grad=True)

p_fun = lambda x: -differentiate(EA_fun(x) * differentiate(u_fun(x), x), x).detach()
g = u_fun(x0).detach()
f = (EA_fun(x1) * differentiate(u_fun(x1), x1)).detach()

x_m = torch.linspace(0, 1, SENSORS).unsqueeze(1)
u_m = u_fun(x_m) + NOISE * torch.randn_like(x_m)

# --------------------------- instantiate model & optimizer ---------------------------
model_EA = MLP(LAYERS, ACTIVATIONS)
model_EA.to(device)
init_weights(model_EA, ACTIVATIONS[0])
EA_hat = lambda x: nn.functional.softplus(model_EA(x))  # positive stiffness
params = list(model_EA.parameters())

if DATA == "full":
    u_hat = u_fun
elif SOLVER == "fem":
    u_hat = lambda x: interpolate(fem(EA_hat, p_fun, g, f, SAMPLES), x)
else:
    model_u = MLP(LAYERS, [nn.Tanh() for _ in range(len(LAYERS) - 2)])
    model_u.to(device)
    init_weights(model_u, ACTIVATIONS[0])
    u_hat = lambda x: model_u(x)
    params += list(model_u.parameters())

residual = lambda x: differentiate(EA_hat(x) * differentiate(u_hat(x), x), x) + p_fun(x)
neumann = lambda: EA_hat(x1) * differentiate(u_hat(x1), x1) - f
term_history = []


def cost_fun():
    if DATA == "full":  # the neumann residual vanishes for any EA, since u'(1)=0
        terms = [pinn_cost([residual(x)])]
    elif SOLVER == "fem":
        terms = [pinn_cost([u_hat(x_m) - u_m])]
    else:
        terms = [
            pinn_cost([residual(x)]),
            pinn_cost([neumann(), u_hat(x0) - g], [1.0, DATA_WEIGHT]),
            pinn_cost([u_hat(x_m) - u_m], [DATA_WEIGHT]),
        ]
    term_history.append([term.item() for term in terms])
    return sum(terms)


# -------------------------------------- training -------------------------------------
tic = time.time()
cost_history = train(cost_fun, params, EPOCHS, LR, decay=DECAY)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

if DATA == "full":  # direct inversion of the integrated equation, EA u' = f + int_x^1 p
    tic = time.time()
    p_w = w * p_fun(x)
    axial_force = f + torch.flip(torch.cumsum(torch.flip(p_w, [0]), 0), [0]) - p_w / 2
    EA_direct = (axial_force / differentiate(u_fun(x), x)).detach()
    toc = time.time()
    error = torch.linalg.norm(EA_direct - EA_fun(x)) / torch.linalg.norm(EA_fun(x))
    print(f"direct inversion: {toc - tic:.2e} s, relative L2 error of EA {error:.2e}")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(0, 1, 200).unsqueeze(1)
EA_test = EA_fun(x_test)
EA_pred_test = EA_hat(x_test).detach()
u_test = u_fun(x_test)
u_pred_test = u_hat(x_test).detach()
error = torch.linalg.norm(EA_pred_test - EA_test) / torch.linalg.norm(EA_test)
print(f"relative L2 error of EA {error:.2e}")

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(cost_history, "k")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test, EA_test, "k")
    ax.plot(x_test, EA_pred_test, "r--")
    plt.show()

    if DATA == "partial":
        fig, ax = plt.subplots()
        ax.plot(x_test, u_test, "k")
        ax.plot(x_m, u_m, "ko")
        ax.plot(x_test, u_pred_test, "r--")
        plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    name = f"bar_inverse_{DATA}" + f"_{SOLVER}" * (DATA == "partial")
    save_csv(
        CSV_DIR / f"{name}.csv",
        x=x_test[:, 0],
        EA=EA_test[:, 0],
        EApred=EA_pred_test[:, 0],
        u=u_test[:, 0],
        upred=u_pred_test[:, 0],
    )
    save_csv(CSV_DIR / f"{name}_sensors.csv", x=x_m[:, 0], u=u_m[:, 0])
    history = {"cost": np.array(cost_history)}
    if DATA == "partial" and SOLVER == "pinn":
        terms = np.array(term_history)
        history.update(Cr=terms[:, 0], Cb=terms[:, 1], Cm=terms[:, 2])
    save_csv(CSV_DIR / f"{name}_cost_history.csv", **history)
