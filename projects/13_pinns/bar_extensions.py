import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from DL import differentiate, init_weights
from NN import MLP
from helper import least_squares, pinn_cost, sample, train
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
OPTIMIZER = "adam"  # adam, lbfgs or elm
SAMPLING = "uniform"  # uniform, random, sobol or adaptive
SELF_ADAPTIVE = False  # learnable weight per point, adam with uniform or sobol only
EPOCHS = 5000 if OPTIMIZER == "adam" else 500
LR = 1e-3 if OPTIMIZER == "adam" else 1.0
SAMPLES = 100  # collocation points
ASCENT_LR = 1e-2  # self-adaptive weights only
REFINE_EVERY = 1000  # adaptive only, epochs between refinements
REFINE_POINTS = 20  # adaptive only, points added per refinement
ELM_SCALE = 10.0  # elm only, range of the random hidden weights and biases

# physics
u_fun = lambda x: torch.cos(2 * torch.pi * x)
EA_fun = lambda x: (torch.cos(2 * torch.pi * x) + 3) / 4

# model settings
LAYERS = [1, 200, 1] if OPTIMIZER == "elm" else [1, 32, 32, 1]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
x, _ = sample([0.0], [1.0], SAMPLES, "uniform" if SAMPLING == "adaptive" else SAMPLING)
x0 = torch.zeros(1, 1, requires_grad=True)
x1 = torch.ones(1, 1, requires_grad=True)

p_fun = lambda x: -differentiate(EA_fun(x) * differentiate(u_fun(x), x), x).detach()
g = u_fun(x0).detach()
f = (EA_fun(x1) * differentiate(u_fun(x1), x1)).detach()

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
init_weights(model, ACTIVATIONS[0])
linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
if OPTIMIZER == "elm":
    for linear in linears[:-1]:
        nn.init.uniform_(linear.weight, -ELM_SCALE, ELM_SCALE)
        nn.init.uniform_(linear.bias, -ELM_SCALE, ELM_SCALE)

u_hat = lambda x: model(x)
dudx = lambda x: differentiate(u_hat(x), x)
residual = lambda x: differentiate(EA_fun(x) * dudx(x), x) + p_fun(x)
pinn_residuals = lambda: [residual(x), u_hat(x0) - g, EA_fun(x1) * dudx(x1) - f]

weights = None
if SELF_ADAPTIVE:
    weights = [torch.ones(len(r), 1, requires_grad=True) for r in pinn_residuals()]
cost_fun = lambda: pinn_cost(pinn_residuals(), weights)


def update_points(epoch):
    global x
    if SAMPLING == "random":
        x, _ = sample([0.0], [1.0], SAMPLES, "random")
    elif SAMPLING == "adaptive" and epoch > 0 and epoch % REFINE_EVERY == 0:
        candidates, _ = sample([0.0], [1.0], 10 * SAMPLES, "random")
        r = residual(candidates).detach().abs().flatten()
        idx = torch.topk(r, REFINE_POINTS).indices
        x = torch.cat([x.detach(), candidates[idx].detach()]).requires_grad_()


# -------------------------------------- training -------------------------------------
tic = time.time()
if OPTIMIZER == "elm":
    least_squares(pinn_residuals, list(linears[-1].parameters()))
    cost_history = []
else:
    cost_history = train(
        cost_fun,
        model.parameters(),
        EPOCHS,
        LR,
        optimizer=OPTIMIZER,
        ascent_params=weights,
        ascent_lr=ASCENT_LR,
        callback=update_points,
    )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(0, 1, 200).unsqueeze(1)
u_test = u_fun(x_test)
u_pred_test = u_hat(x_test).detach()
error = torch.linalg.norm(u_pred_test - u_test) / torch.linalg.norm(u_test)
print(f"relative L2 error {error:.2e}")
x_points = x.detach()
u_points = u_hat(x_points).detach()

if not args.book:
    if cost_history:
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.plot(cost_history, "k")
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test, u_test, "k")
    ax.plot(x_test, u_pred_test, "r--")
    ax.plot(x_points, u_points, "r.")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    name = f"bar_extensions_{OPTIMIZER}_{SAMPLING}" + "_selfadaptive" * SELF_ADAPTIVE
    save_csv(
        CSV_DIR / f"{name}.csv",
        x=x_test[:, 0],
        u=u_test[:, 0],
        upred=u_pred_test[:, 0],
    )
    save_csv(CSV_DIR / f"{name}_points.csv", x=x_points[:, 0], upred=u_points[:, 0])
    if cost_history:
        save_csv(CSV_DIR / f"{name}_cost_history.csv", cost=np.array(cost_history))
