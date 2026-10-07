import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from DL import differentiate, init_weights
from NN import MLP
from helper import energy_cost, fem, grid, interpolate, train
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small
torch.set_num_threads(1)  # threading overhead dominates for such small networks

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
METHOD = "hidenn"  # fem, hidenn or feinn
OPTIMIZER = "lbfgs"  # adam or lbfgs
EPOCHS = 5000 if OPTIMIZER == "adam" else 20  # each l-bfgs epoch runs up to 20 iterations
LR = 1e-2 if OPTIMIZER == "adam" else 1.0
ELEMENTS = 100  # linear elements, one midpoint each for the quadrature

# physics
u_fun = lambda x: torch.cos(2 * torch.pi * x)
EA_fun = lambda x: (torch.cos(2 * torch.pi * x) + 3) / 4

# model settings
LAYERS = [1, 32, 32, 1]
ACTIVATIONS = [nn.GELU() for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
x, w = grid([0.0], [1.0], ELEMENTS)  # element midpoints, exact for linear elements
x_nodes = torch.linspace(0, 1, ELEMENTS + 1)[1:].unsqueeze(1)  # without x=0
x0 = torch.zeros(1, 1, requires_grad=True)
x1 = torch.ones(1, 1, requires_grad=True)

p_fun = lambda x: -differentiate(EA_fun(x) * differentiate(u_fun(x), x), x).detach()
g = u_fun(x0).detach()
f = (EA_fun(x1) * differentiate(u_fun(x1), x1)).detach()

x_test = torch.linspace(0, 1, 200).unsqueeze(1)
u_test = u_fun(x_test)
relative_error = lambda u: (torch.linalg.norm(u - u_test) / torch.linalg.norm(u_test)).item()

# --------------------------- instantiate model & optimizer ---------------------------
if METHOD == "hidenn":  # the nodal values themselves are the parameters
    u_free = nn.Parameter(torch.zeros(ELEMENTS))
    params = [u_free]
    u_nodes = lambda: torch.cat([g[0], u_free])
elif METHOD == "feinn":  # a network predicts the nodal values
    model = MLP(LAYERS, ACTIVATIONS)
    model.to(device)
    init_weights(model, ACTIVATIONS[0])
    params = list(model.parameters())
    u_nodes = lambda: torch.cat([g[0], model(x_nodes)[:, 0]])

# hard-coded linear shape functions, strains by automatic differentiation
u_hat = lambda x: interpolate(u_nodes(), x)
dudx = lambda x: differentiate(u_hat(x), x)
cost_fun = lambda: energy_cost(
    [
        (x, w, lambda x: 0.5 * EA_fun(x) * dudx(x) ** 2 - p_fun(x) * u_hat(x)),
        (x1, 1, lambda x: -f * u_hat(x)),
    ]
)

# -------------------------------------- training -------------------------------------
tic = time.time()
if METHOD == "fem":
    u_fem = fem(EA_fun, p_fun, g, f, ELEMENTS).detach()
    u_hat = lambda x: interpolate(u_fem, x)
    cost_history = []
else:
    cost_history = train(cost_fun, params, EPOCHS, LR, optimizer=OPTIMIZER)
toc = time.time()
print(f"elapsed time {toc - tic:.2e} s")

u_pred_test = u_hat(x_test).detach()
print(f"relative L2 error {relative_error(u_pred_test):.2e}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    if cost_history:
        fig, ax = plt.subplots()
        ax.plot(cost_history, "k")
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test, u_test, "k")
    ax.plot(x_test, u_pred_test, "r--")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / f"bar_ansatz_{METHOD}.csv",
        x=x_test[:, 0],
        u=u_test[:, 0],
        upred=u_pred_test[:, 0],
    )
