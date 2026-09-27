import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from DL import init_weights
from NN import MLP
from helper import (
    energy_cost,
    energy_error,
    plane_stress,
    plate_points,
    plate_potential,
    strain,
    train,
)
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
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
RESOLUTIONS = [40, 80, 160]  # cells per side, multiples of 5 * holes
TEST_RESOLUTION = 640

# physics
E = 1.0
NU = 0.3
TRACTION = 1.0
HOLES = [1, 2, 4, 8]  # holes per side of the perforated plate

# model settings
LAYERS = [2, 64, 64, 64, 2]

# finite element comparison
DEGREE = 2  # at the element counts RESOLUTIONS

# ------------------------------------ geometry study ---------------------------------
reference = np.load(DATA_DIR / "elasticity2D_reference.npz")
C = plane_stress(E, NU).to(device)
t_right = torch.tensor([[TRACTION, 0.0]], device=device)

results = {"holes": [], "resolution": [], "error": [], "time": []}
for holes in HOLES:
    compliance = reference["compliance"][list(reference["holes"]).index(holes)]
    for resolution in RESOLUTIONS:
        x, w, x_right, w_right = plate_points(holes, resolution)
        x, w = x.detach().to(device).requires_grad_(), w.to(device)
        x_right, w_right = x_right.detach().to(device), w_right.to(device)

        activations = [nn.Tanh() for _ in range(len(LAYERS) - 2)]
        model = MLP(LAYERS, activations)
        model.to(device)
        init_weights(model, activations[0])

        u_hat = lambda x: x[:, 0:1] * model(x)

        def density(x):
            epsilon = strain(u_hat(x), x)
            return 0.5 * torch.sum(epsilon * (epsilon @ C), 1, keepdim=True)

        cost_fun = lambda: energy_cost(
            [(x, w, density), (x_right, w_right, lambda x: -u_hat(x) @ t_right.T)]
        )

        tic = time.time()
        train(cost_fun, model.parameters(), EPOCHS, LR)
        toc = time.time()

        potential = plate_potential(u_hat, C, t_right, holes, TEST_RESOLUTION, device)
        error = energy_error(potential, compliance)
        print(f"{holes} holes resolution {resolution} error {error:.2e}")
        for key, value in zip(results, [holes, resolution, error, toc - tic]):
            results[key].append(value)
results = {key: np.array(value) for key, value in results.items()}

# ----------------------------------- postprocessing ----------------------------------
study = {key[6:]: reference[key] for key in reference.files if key.startswith("study")}
compliances = reference["compliance"][
    np.searchsorted(reference["holes"], study["holes"])
]
study["error"] = np.sqrt((compliances - study["compliance"]) / compliances)

fig, ax = plt.subplots()
for resolution in RESOLUTIONS:
    mask = results["resolution"] == resolution
    ax.loglog(results["holes"][mask], results["error"][mask], "r.-")
    mask = (study["elements"] == resolution) & (study["degree"] == DEGREE)
    ax.loglog(study["holes"][mask], study["error"][mask], "b.--")
ax.set_xlabel("holes per side")
ax.set_ylabel("relative energy error")

geometries = []
for holes in HOLES:
    fig_geometry, ax_geometry = plt.subplots(figsize=(4, 4))
    field = reference[f"ux_{holes}"]
    mesh = ax_geometry.pcolormesh(reference["x"], reference["y"], field, cmap="turbo")
    ax_geometry.set_aspect("equal")
    ax_geometry.axis("off")
    mesh.set_rasterized(True)
    fig_geometry.subplots_adjust(left=0, right=1, top=1, bottom=0)
    geometries.append((fig_geometry, f"elasticity2D_geometry_{holes}.pdf"))

if not args.book:
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for fig_geometry, name in geometries:
        fig_geometry.savefig(RGB_PDF_DIR / name)
    for resolution in RESOLUTIONS:
        mask = results["resolution"] == resolution
        save_csv(
            CSV_DIR / f"elasticity2D_geometry_dem_{resolution}.csv",
            holes=results["holes"][mask],
            error=results["error"][mask],
            time=results["time"][mask],
        )
        mask = (study["elements"] == resolution) & (study["degree"] == DEGREE)
        save_csv(
            CSV_DIR / f"elasticity2D_geometry_fem_{resolution}.csv",
            holes=study["holes"][mask],
            error=study["error"][mask],
            time=study["time"][mask],
        )
