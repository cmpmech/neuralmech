import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from DL import differentiate, init_weights
from NN import MLP
from helper import (
    energy_cost,
    fem_reference,
    gauss_grid,
    graded_quadrature,
    grid,
    pinn_cost,
    train,
    weak_cost,
)
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
METHOD = "pinn"  # pinn, dem, vpinn or wan
PROBLEM = "smooth"  # smooth or singular
VALIDATION = False  # dem only, energy on a finer grid to reveal overfitting
EPOCHS = 5500 if VALIDATION else 5000  # longer, so that the overfitting fully develops
LR = 3e-3 if METHOD == "pinn" and PROBLEM == "smooth" else 1e-3
DECAY = 0.999 if METHOD == "wan" else 1.0  # learning rate factor per epoch
SAMPLES = 100  # collocation and quadrature points
ORDER = 4  # gauss points per cell of the integral methods
QUADRATURE = "uniform"  # uniform or graded toward x=0 (singular problem)
TEST_FUNCTIONS = 20  # vpinn only
DEGREE = 2  # polynomial degree of the finite element reference
DOFS = 150  # finite element reference
ASCENT_LR = 1e-2  # wan only, test function learning rate

# physics
if PROBLEM == "smooth":
    u_fun = lambda x: torch.cos(2 * torch.pi * x)
    EA_fun = lambda x: (torch.cos(2 * torch.pi * x) + 3) / 4
else:
    u_fun = lambda x: x**0.65 - 0.65 * x  # weak singularity at x=0
    EA_fun = lambda x: torch.ones_like(x)

# model settings
LAYERS = [1, 32, 32, 1]
# auto: gelu trains best on the smooth bar, while tanh suits the singularity and reveals
# the quadrature overfitting that gelu avoids; tanh also lets the wan collapse
ACTIVATION = "auto"  # auto, gelu or tanh
DEFAULT = "gelu" if PROBLEM == "smooth" and not VALIDATION else "tanh"
ACTIVATION = DEFAULT if ACTIVATION == "auto" else ACTIVATION
activation = {"gelu": nn.GELU, "tanh": nn.Tanh}[ACTIVATION]
ACTIVATIONS = [activation() for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
if QUADRATURE == "graded":  # gauss points on cells graded toward the singularity
    x, w = graded_quadrature(1, 10, ORDER)
    x.requires_grad_()
elif METHOD == "pinn":
    x, w = grid([0.0], [1.0], SAMPLES)  # midpoint collocation points
else:
    x, w = gauss_grid([0.0], [1.0], SAMPLES // ORDER, ORDER)
x_val, w_val = gauss_grid([0.0], [1.0], 10 * SAMPLES // ORDER, ORDER)
x0 = torch.zeros(1, 1, requires_grad=True)
x1 = torch.ones(1, 1, requires_grad=True)

p_fun = lambda x: -differentiate(EA_fun(x) * differentiate(u_fun(x), x), x).detach()
g = u_fun(x0).detach()
f = (EA_fun(x1) * differentiate(u_fun(x1), x1)).detach()

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
init_weights(model, ACTIVATIONS[0])
PARAMETERS = sum(param.numel() for param in model.parameters())
print(f"network parameters {PARAMETERS}")

if METHOD == "pinn":
    u_hat = lambda x: model(x)
else:
    u_hat = lambda x: g + x * model(x)  # strong Dirichlet boundary condition

dudx = lambda x: differentiate(u_hat(x), x)
ascent_params = None
term_history = []

if METHOD == "pinn":
    residual = lambda x: differentiate(EA_fun(x) * dudx(x), x) + p_fun(x)
    cost_r = lambda: pinn_cost([residual(x)])
    cost_b = lambda: pinn_cost([u_hat(x0) - g, EA_fun(x1) * dudx(x1) - f])

    def cost_fun():
        terms = [cost_r(), cost_b()]
        term_history.append([term.item() for term in terms])
        return sum(terms)

elif METHOD == "dem":
    internal = lambda x, w: energy_cost([(x, w, lambda x: 0.5 * EA_fun(x) * dudx(x) ** 2)])
    external = lambda x, w: energy_cost(
        [(x, w, lambda x: -p_fun(x) * u_hat(x)), (x1, 1, lambda x: -f * u_hat(x))]
    )
    energy = lambda x, w: internal(x, w) + external(x, w)

    def cost_fun():
        terms = [internal(x, w), external(x, w)]
        term_history.append([term.item() for term in terms])
        return sum(terms)
else:
    weak_form = [
        (x, w, lambda x, v, dv: -EA_fun(x) * dudx(x) * dv[..., 0] + p_fun(x) * v),
        (x1, 1, lambda x, v, dv: f * v),
    ]
    if METHOD == "vpinn":
        k = torch.arange(1, TEST_FUNCTIONS + 1)
        test_functions = lambda x: torch.sin((k - 0.5) * torch.pi * x)
        cost_fun = lambda: weak_cost(weak_form, test_functions)
    else:
        model_v = MLP(LAYERS, [activation() for _ in range(len(LAYERS) - 2)])
        model_v.to(device)
        init_weights(model_v, ACTIVATIONS[0])
        test_functions = lambda x: x * model_v(x)
        cost_fun = lambda: weak_cost(weak_form, test_functions, normalize=True)
        ascent_params = model_v.parameters()

if PROBLEM == "smooth":
    x_test = torch.linspace(0, 1, 200).unsqueeze(1)
else:
    x_test = torch.logspace(-4, 0, 200).unsqueeze(1)  # resolves the singularity
u_test = u_fun(x_test)
relative_error = lambda u: (torch.linalg.norm(u - u_test) / torch.linalg.norm(u_test)).item()
val_history = []
error_history = []


def validate(epoch):
    if VALIDATION:
        val_history.append(energy(x_val, w_val).item())
    if epoch % 10 == 0:  # sparse, since the evaluation slows down training
        with torch.no_grad():
            error_history.append([epoch, relative_error(u_hat(x_test))])


# -------------------------------------- training -------------------------------------
tic = time.time()
cost_history = train(
    cost_fun,
    model.parameters(),
    EPOCHS,
    LR,
    decay=DECAY,
    ascent_params=ascent_params,
    ascent_lr=ASCENT_LR,
    callback=validate,
)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

u_pred_test = u_hat(x_test).detach()
error = relative_error(u_pred_test)
print(f"relative L2 error {error:.2e}")


u_double = u_fun(x_test.double())  # double, since the reference error is tiny
error_double = lambda u: (torch.linalg.norm(u - u_double) / torch.linalg.norm(u_double)).item()
x1_double = x1.double().detach().requires_grad_()
f_double = (EA_fun(x1_double) * differentiate(u_fun(x1_double), x1_double)).detach()


def fem_error(dofs):  # quadratic elements, two dofs per element after dirichlet
    tic = time.time()
    u_fem = fem_reference(EA_fun, p_fun, g, f_double, dofs // DEGREE, DEGREE)(x_test)
    toc = time.time()
    return toc - tic, error_double(u_fem)


lower, upper = 1, PARAMETERS // DEGREE
while lower < upper:  # bisect the fewest elements that match the network error
    middle = (lower + upper) // 2
    lower, upper = (lower, middle) if fem_error(DEGREE * middle)[1] <= error else (middle + 1, upper)
for dofs in (DOFS, DEGREE * lower, DEGREE * (PARAMETERS // DEGREE)):
    elapsed, fem_err = fem_error(dofs)
    print(f"finite elements, {dofs} dofs: {elapsed:.2e} s, relative L2 error {fem_err:.2e}")


# ----------------------------------- postprocessing ----------------------------------
x_test.requires_grad_()
dudx_test = differentiate(u_fun(x_test), x_test).detach()
dudx_pred_test = dudx(x_test).detach()
x_test = x_test.detach()

if not args.book:
    fig, ax = plt.subplots()
    if METHOD != "dem":
        ax.set_yscale("log")
    ax.plot(cost_history, "k")
    if VALIDATION:
        ax.plot(val_history, "r--")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_test, u_test, "k")
    ax.plot(x_test, u_pred_test, "r--")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    name = f"bar_forward_{METHOD}" + "_singular" * (PROBLEM == "singular")
    name += "_validation" * VALIDATION + f"_{ACTIVATION}" * (ACTIVATION != DEFAULT)
    name += "_graded" * (QUADRATURE == "graded")
    save_csv(
        CSV_DIR / f"{name}.csv",
        x=x_test[:, 0],
        u=u_test[:, 0],
        upred=u_pred_test[:, 0],
        dudx=dudx_test[:, 0],
        dudxpred=dudx_pred_test[:, 0],
    )
    x_points = x.detach()
    u_points = u_hat(x_points).detach()
    save_csv(CSV_DIR / f"{name}_points.csv", x=x_points[:, 0], upred=u_points[:, 0])
    history = {"cost": np.array(cost_history)}
    if VALIDATION:
        history["val"] = np.array(val_history)
    if METHOD == "pinn":
        terms = np.array(term_history)
        history.update(Cr=terms[:, 0], Cb=terms[:, 1])
    elif METHOD == "dem":
        terms = np.array(term_history)
        history.update(internal=terms[:, 0], external=terms[:, 1])
    save_csv(CSV_DIR / f"{name}_cost_history.csv", **history)
    errors = np.array(error_history)
    save_csv(CSV_DIR / f"{name}_error_history.csv", epoch=errors[:, 0], error=errors[:, 1])
