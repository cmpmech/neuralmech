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
    energy_error,
    face,
    hole_faces,
    pinn_cost,
    plane_stress,
    plate_points,
    plate_potential,
    strain,
    traction,
    train,
)
from postprocessing import load_cmap, save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
METHOD = "dem"  # pinn or dem
EPOCHS = 5000
LR = 1e-3
RESOLUTION = 50  # cells per side, multiple of 5 * HOLES, so the holes lie on cell faces
TEST_RESOLUTION = 640  # grid for the potential energy of the prediction

# physics
E = 1.0
NU = 0.3
TRACTION = 1.0
HOLES = 1  # holes per side of the perforated plate

# model settings
LAYERS = [2, 64, 64, 64, 2]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]

# ------------------------------------ prepare data -----------------------------------
C = plane_stress(E, NU).to(device)
t_right = torch.tensor([[TRACTION, 0.0]], device=device)

x, w, x_right, w_right = plate_points(HOLES, RESOLUTION)
x, w = x.detach().to(device).requires_grad_(), w.to(device)
x_right, w_right = x_right.detach().to(device).requires_grad_(), w_right.to(device)

traction_free = [face([0.0, 0.0], [1.0, 1.0], RESOLUTION, 1, side) for side in [0, 1]]
traction_free += hole_faces(HOLES, RESOLUTION)
traction_free = [
    (x_b.detach().to(device).requires_grad_(), n_b.to(device))
    for x_b, _, n_b in traction_free
]

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
init_weights(model, ACTIVATIONS[0])

u_hat = lambda x: x[:, 0:1] * model(x)  # clamped left edge
epsilon = lambda x: strain(u_hat(x), x)
sigma = lambda x: epsilon(x) @ C

if METHOD == "pinn":

    def cost_fun():
        s = sigma(x)
        div_x = differentiate(s[:, 0:1], x)[:, 0] + differentiate(s[:, 2:3], x)[:, 1]
        div_y = differentiate(s[:, 2:3], x)[:, 0] + differentiate(s[:, 1:2], x)[:, 1]
        n_right = torch.tensor([[1.0, 0.0]], device=device)
        residuals = [div_x, div_y, traction(sigma(x_right), n_right) - t_right]
        for x_b, n_b in traction_free:
            residuals.append(traction(sigma(x_b), n_b))
        return pinn_cost(residuals)

else:
    cost_fun = lambda: energy_cost(
        [
            (x, w, lambda x: 0.5 * torch.sum(epsilon(x) * sigma(x), 1, keepdim=True)),
            (x_right, w_right, lambda x: -u_hat(x) @ t_right.T),
        ]
    )

# -------------------------------------- training -------------------------------------
tic = time.time()
cost_history = train(cost_fun, model.parameters(), EPOCHS, LR)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
reference = np.load(DATA_DIR / "elasticity2D_reference.npz")
compliance = reference["compliance"][list(reference["holes"]).index(HOLES)]
potential = plate_potential(u_hat, C, t_right, HOLES, TEST_RESOLUTION, device)
error = energy_error(potential, compliance)
print(f"relative energy error {error:.2e}")

study = {key[6:]: reference[key] for key in reference.files if key.startswith("study")}
study_errors = np.sqrt((compliance - study["compliance"]) / compliance)
study_errors[study["holes"] != HOLES] = np.inf
closest = np.argmin(np.abs(np.log(study_errors / error)))
coarse = f"{HOLES}_{study['elements'][closest]}_{study['degree'][closest]}"
print(
    f"finite elements with a comparable error: {study['elements'][closest]} elements "
    f"per side, p={study['degree'][closest]}, error {study_errors[closest]:.2e}"
)

x_test = np.stack([reference["x"], reference["y"]], axis=-1).reshape(-1, 2)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device).requires_grad_()
u_pred = u_hat(x_test)
exx_pred = differentiate(u_pred[:, 0:1], x_test)[:, 0]
shape = reference["x"].shape
predictions = {
    "ux": u_pred[:, 0].detach().cpu().numpy().reshape(shape),
    "exx": exx_pred.detach().cpu().numpy().reshape(shape),
}
in_hole = np.isnan(reference[f"ux_{HOLES}"])
predictions = {
    key: np.where(in_hole, np.nan, value) for key, value in predictions.items()
}

rows = {
    "reference": {key: reference[f"{key}_{HOLES}"] for key in ["ux", "exx"]},
    METHOD: predictions,
    "coarse": {key: reference[f"{key}_{coarse}"] for key in ["ux", "exx"]},
}
figures = []
for row, fields in rows.items():
    for key, cmap in [("ux", "turbo"), ("exx", rainbow)]:
        fig, ax = plt.subplots(figsize=(4, 4))
        mesh = ax.pcolormesh(reference["x"], reference["y"], fields[key], cmap=cmap)
        limits = rows["reference"][key]
        if key == "exx":
            mesh.set_clim(*np.nanpercentile(limits, [1, 99]))
        else:
            mesh.set_clim(np.nanmin(limits), np.nanmax(limits))
        ax.set_aspect("equal")
        ax.axis("off")
        mesh.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figures.append((fig, f"elasticity2D_{row}_{key}.pdf"))

if not args.book:
    fig, ax = plt.subplots()
    if METHOD == "pinn":
        ax.set_yscale("log")
    ax.plot(cost_history, "k")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for fig, name in figures:
        fig.savefig(RGB_PDF_DIR / name)
    save_csv(
        CSV_DIR / f"elasticity2D_{METHOD}_cost_history.csv", cost=np.array(cost_history)
    )
    save_csv(
        CSV_DIR / f"elasticity2D_{METHOD}_errors.csv",
        network=np.array([error]),
        fem=np.array([study_errors[closest]]),
        elements=np.array([study["elements"][closest]]),
        degree=np.array([study["degree"][closest]]),
    )
