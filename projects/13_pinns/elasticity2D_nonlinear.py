import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from DL import differentiate, init_weights
from NN import MLP
from helper import energy_cost, plate_points, train
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 5000
LR = 1e-3
RESOLUTION = 50

# physics
E = 1.0
NU = 0.3
HOLES = 1

# model settings
LAYERS = [2, 64, 64, 64, 2]

# --------------------------------------- helper --------------------------------------
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


def neo_hookean(u, x):
    dux = differentiate(u[:, 0:1], x)
    duy = differentiate(u[:, 1:2], x)
    F11, F12 = 1 + dux[:, 0:1], dux[:, 1:2]
    F21, F22 = duy[:, 0:1], 1 + duy[:, 1:2]
    J = F11 * F22 - F12 * F21
    log_J = torch.log(torch.clamp(J, min=1e-6))  # an inverted element has no energy
    trace_C = F11**2 + F12**2 + F21**2 + F22**2
    return 0.5 * MU * (trace_C - 2 - 2 * log_J) + 0.5 * LAMBDA * log_J**2


# ---------------------------------- nonlinearity study -------------------------------
reference = np.load(DATA_DIR / "elasticity2D_nonlinear_reference.npz")
x_test = np.stack([reference["x"], reference["y"]], axis=-1).reshape(-1, 2)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)

x, w, x_right, w_right = plate_points(HOLES, RESOLUTION)
x, w = x.detach().to(device).requires_grad_(), w.to(device)
x_right, w_right = x_right.detach().to(device), w_right.to(device)

results = {"error": [], "time": []}
for traction in reference["traction"]:
    t_right = torch.tensor([[float(traction), 0.0]], device=device)

    activations = [nn.Tanh() for _ in range(len(LAYERS) - 2)]
    model = MLP(LAYERS, activations)
    model.to(device)
    init_weights(model, activations[0])

    u_hat = lambda x: float(traction) * x[:, 0:1] * model(x)  # network predicts u / t
    cost_fun = lambda: energy_cost(
        [
            (x, w, lambda x: neo_hookean(u_hat(x), x)),
            (x_right, w_right, lambda x: -u_hat(x) @ t_right.T),
        ]
    )

    tic = time.time()
    train(cost_fun, model.parameters(), EPOCHS, LR)
    toc = time.time()

    u_ref = np.stack([reference[f"ux_{traction}"], reference[f"uy_{traction}"]], -1)
    u_pred = u_hat(x_test).detach().cpu().numpy().reshape(u_ref.shape)
    error = np.sqrt(np.nansum((u_pred - u_ref) ** 2) / np.nansum(u_ref**2))
    print(f"traction {traction} error {error:.2e}")
    results["error"].append(error)
    results["time"].append(toc - tic)
results = {key: np.array(value) for key, value in results.items()}

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].loglog(reference["nonlinearity"], results["error"], "r.-")
    axs[0].loglog(reference["nonlinearity"], reference["error"], "b.--")
    axs[0].set_ylabel("relative displacement error")
    axs[1].semilogx(reference["nonlinearity"], results["time"], "r.-")
    axs[1].semilogx(reference["nonlinearity"], reference["time"], "b.--")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("time in s")
    for ax in axs:
        ax.set_xlabel("nonlinearity")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "elasticity2D_nonlinear_dem.csv",
        nonlinearity=reference["nonlinearity"],
        error=results["error"],
        time=results["time"],
    )
    save_csv(
        CSV_DIR / "elasticity2D_nonlinear_fem.csv",
        nonlinearity=reference["nonlinearity"],
        error=reference["error"],
        time=reference["time"],
    )
