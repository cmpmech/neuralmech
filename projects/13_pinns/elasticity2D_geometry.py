import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.func import jvp

from DL import init_weights
from NN import MLP
from helper import energy_cost, energy_error, face, gyroid_mask, spacetree_quadrature, strain, train
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
# geometric complexity, gyroid unit cells per side, any of the reference (1 to 12 in
# steps of 0.5); one value shows its fields, several run the study
CELLS = [2.0]
if args.book:  # the study of the book figure, all geometries of the reference
    CELLS = np.round(np.arange(1.0, 12.01, 0.5), 1).tolist()

# hyperparameters
EPOCHS = 3000  # after the extreme learning machine initialization
LR = 1e-3
DECAY = 0.9998

# quadrature, space tree on the implicit geometry as in the finite cell method
RESOLUTIONS = [128]  # cells per side
DEPTH = 2  # levels of subdivision of the cut cells
ORDER = 2  # Gauss points per direction and leaf
TEST_RESOLUTION, TEST_DEPTH, TEST_ORDER = 320, 4, 3  # potential energy after training

# physics, plane stress, clamped on the left, loaded downwards on the right edge
E = 1.0
NU = 0.3
TRACTION = [0.0, -1.0]
LAM, MU = E * NU / (1 - NU**2), 0.5 * E / (1 + NU)

# model settings
FEATURES = 64  # random Fourier features
SIGMA = 2.0  # best of 1, 2, 4 at 2 cells (energy error 12, 10, 13 %)
LAYERS = [2 * FEATURES, 128, 128, 128, 128, 2]

# ------------------------------------- load data -------------------------------------
reference = np.load(DATA_DIR / "elasticity2D_geometry_reference.npz")
R = round(math.sqrt(reference["u"].shape[1]))  # evaluation grid of the reference
axis = (np.arange(R) + 0.5) / R
x_eval = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)


# --------------------------------------- helper --------------------------------------
def density(x):
    """strain energy density, plane stress."""
    epsilon = strain(u_hat(x), x)
    trace = epsilon[:, 0] + epsilon[:, 1]
    squared = epsilon[:, 0] ** 2 + epsilon[:, 1] ** 2 + 0.5 * epsilon[:, 2] ** 2
    return (0.5 * LAM * trace**2 + MU * squared)[:, None]


def last_layer_basis(x):
    """g = x_0 [h, 1] with the hidden features h of the network and its x and y
    derivatives, u = g W for the last layer W (with its bias)."""
    ones, zeros = torch.ones(x.shape[0], 1, device=device), torch.zeros(x.shape[0], 1, device=device)
    hidden = lambda x: model.model[:-1](embed(x))
    h, h_x = jvp(hidden, (x,), (torch.cat([ones, zeros], 1),))
    _, h_y = jvp(hidden, (x,), (torch.cat([zeros, ones], 1),))
    h, h_x, h_y = torch.cat([h, ones], 1), torch.cat([h_x, zeros], 1), torch.cat([h_y, zeros], 1)
    return x[:, 0:1] * h, h + x[:, 0:1] * h_x, x[:, 0:1] * h_y


def initialize_last_layer(chunk=8192):
    """extreme learning machine: with the hidden layers fixed, the potential energy is
    quadratic in the last layer, so its minimizer solves K w = f."""
    blocks = None
    for x_chunk, w_chunk in zip(x.detach().split(chunk), w.reshape(-1).split(chunk)):
        with torch.no_grad():
            _, g_x, g_y = last_layer_basis(x_chunk)
        g_x, g_y = g_x.double(), g_y.double()
        a, b, c = [(w_chunk * k).double()[:, None] for k in (LAM + 2 * MU, MU, LAM)]
        terms = [
            (a * g_x).T @ g_x + (b * g_y).T @ g_y,  # u_x, u_x
            (c * g_x).T @ g_y + (b * g_y).T @ g_x,  # u_x, u_y
            (a * g_y).T @ g_y + (b * g_x).T @ g_x,  # u_y, u_y
        ]
        blocks = terms if blocks is None else [B + T for B, T in zip(blocks, terms)]
    K = torch.cat([torch.cat(blocks[:2], 1), torch.cat([blocks[1].T, blocks[2]], 1)])
    with torch.no_grad():
        g, _, _ = last_layer_basis(x_right)
    f = (w_right.reshape(-1, 1) * g).sum(0).double()
    f = torch.cat([TRACTION[0] * f, TRACTION[1] * f])
    # minimum-norm solution, the random features are nearly linearly dependent
    eigenvalues, V = torch.linalg.eigh(K)
    keep = eigenvalues > 1e-12 * eigenvalues.max()
    solution = (V[:, keep] @ ((V[:, keep].T @ f) / eigenvalues[keep])).float().reshape(2, -1)
    with torch.no_grad():
        model.model[-1].weight.copy_(solution[:, :-1])
        model.model[-1].bias.copy_(solution[:, -1])


def potential(cells):
    """potential energy of the network on a finer quadrature, in chunks, as the network
    may fit the quadrature it was trained on."""
    inside = lambda x: gyroid_mask(x, cells)
    x, w = spacetree_quadrature(inside, TEST_RESOLUTION, TEST_DEPTH, TEST_ORDER)
    x_right, w_right, _ = face([0.0, 0.0], [1.0, 1.0], 8 * TEST_RESOLUTION, 0, 1)
    with torch.no_grad():
        u_right = u_hat(x_right.detach().to(device))
    energy = -torch.sum(w_right.to(device) * (u_right @ t.T)).item()
    for x_chunk, w_chunk in zip(x.detach().split(2**16), w.split(2**16)):
        x_chunk = x_chunk.to(device).requires_grad_()
        energy += torch.sum(w_chunk.to(device) * density(x_chunk)).item()
    return energy


# ---------------------------------- geometry study -----------------------------------
t = torch.tensor([TRACTION], device=device)
results = {resolution: {"cells": [], "error": [], "energy": [], "time": []} for resolution in RESOLUTIONS}
for cells in CELLS:
    index = list(np.round(reference["cells"], 1)).index(round(cells, 1))
    u_ref = reference["u"][index]
    solid = ~np.isnan(u_ref[:, 0])
    for resolution in RESOLUTIONS:
        torch.manual_seed(0)
        x, w = spacetree_quadrature(lambda x: gyroid_mask(x, cells), resolution, DEPTH, ORDER)
        x, w = x.detach().to(device).requires_grad_(), w.to(device)
        x_right, w_right, _ = face([0.0, 0.0], [1.0, 1.0], 2 * resolution, 0, 1)
        x_right, w_right = x_right.detach().to(device), w_right.to(device)

        B = (torch.randn(2, FEATURES) * SIGMA).to(device)
        activations = [nn.Tanh() for _ in range(len(LAYERS) - 2)]
        model = MLP(LAYERS, activations)
        model.to(device)
        init_weights(model, activations[0])
        embed = lambda x: torch.cat([torch.sin(2 * math.pi * x @ B), torch.cos(2 * math.pi * x @ B)], 1)
        u_hat = lambda x: x[:, 0:1] * model(embed(x))  # clamped, u = 0 at x = 0
        cost_fun = lambda: energy_cost(
            [(x, w, density), (x_right, w_right, lambda x: -u_hat(x) @ t.T)]
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.time()
        initialize_last_layer()
        cost_history = train(cost_fun, model.parameters(), EPOCHS, LR, decay=DECAY)
        if device.type == "cuda":
            torch.cuda.synchronize()
        toc = time.time()

        with torch.no_grad():
            u = u_hat(torch.tensor(x_eval[solid], dtype=torch.float32, device=device)).cpu().numpy()
        error = np.linalg.norm(u - u_ref[solid]) / np.linalg.norm(u_ref[solid])
        energy = energy_error(potential(cells), reference["compliance"][index])
        print(
            f"{cells:g} cells, {resolution} quadrature cells ({len(x)} points): "
            f"time {toc - tic:.1f} s, displacement error {error:.2e}, energy error {energy:.2e}"
        )
        for key, value in zip(results[resolution], [cells, error, energy, toc - tic]):
            results[resolution][key].append(value)

# ----------------------------------- postprocessing ----------------------------------
study = {key[6:]: reference[key] for key in reference.files if key.startswith("study_")}
fields = [key for key in reference.files if key.startswith("field_")]

figures = []
for key in fields:
    field = reference[key]
    resolution = round(math.sqrt(len(field)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(field[:, 1].reshape(resolution, resolution).T, origin="lower", extent=(0, 1, 0, 1), cmap="turbo")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figures.append((fig, f"elasticity2D_geometry_{key[6:]}.png"))

if not args.book:
    if len(CELLS) == 1:
        u_field = np.full_like(u_ref, np.nan)
        u_field[solid] = u
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax, value, title in zip(
            axs,
            [u_ref[:, 1], u_field[:, 1], np.linalg.norm(u_field - u_ref, axis=1)],
            ["reference $u_y$", "deep energy method $u_y$", "displacement error"],
        ):
            image = ax.imshow(value.reshape(R, R).T, origin="lower", extent=(0, 1, 0, 1), cmap="hot_r" if "error" in title else "turbo")
            fig.colorbar(image, ax=ax)
            ax.set_title(title)
            ax.axis("off")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for resolution in RESOLUTIONS:
            axs[0].semilogy(results[resolution]["cells"], results[resolution]["error"], ".-", label=f"DEM {resolution}")
            axs[1].semilogy(results[resolution]["cells"], results[resolution]["time"], ".-")
        for degree, elements in sorted(set(zip(study["degree"], study["elements"]))):
            mask = (study["degree"] == degree) & (study["elements"] == elements)
            axs[0].semilogy(study["cells"][mask], study["error"][mask], "--", label=f"FEM p={degree} {elements}")
            axs[1].semilogy(study["cells"][mask], study["time"][mask], "--")
        axs[0].set_ylabel("relative displacement error")
        axs[1].set_ylabel("time [s]")
        for ax in axs:
            ax.set_xlabel("unit cells per side")
        axs[0].legend()
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for fig, name in figures:
        fig.savefig(RGB_PDF_DIR / name, dpi=128)
    for resolution in RESOLUTIONS:
        save_csv(CSV_DIR / f"elasticity2D_geometry_dem_{resolution}.csv", **results[resolution])
    for degree, elements in sorted(set(zip(study["degree"], study["elements"]))):
        mask = (study["degree"] == degree) & (study["elements"] == elements)
        save_csv(
            CSV_DIR / f"elasticity2D_geometry_fem_p{degree}_{elements}.csv",
            cells=study["cells"][mask],
            error=study["error"][mask],
            ndof=study["ndof"][mask],
            time=study["time"][mask],
        )
