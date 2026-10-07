import argparse
import math
import sys
import time
import tomllib
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torch import nn

from DL import init_weights
from NN import MLP
from helper import energy_cost, face, gauss_grid, strain, train
from postprocessing import load_cmap, save_csv

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "../18_voxel_mlhp"))  # matrix-free voxel finite elements

from voxel import VoxelMultigrid

BENCHMARK_DIR = (BASE_DIR / "../1_benchmark_2D").resolve()
DATA_DIR = (BASE_DIR / "../../data/2D_benchmark").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

with open(BENCHMARK_DIR / "settings.toml", "rb") as f:
    MATERIAL = tomllib.load(f)["materials"]["elasticity"]

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 15000
LR = 1e-3
DECAY = 0.9998

# geometry, the first sample of the Chapter 1 benchmark at 256^2, thresholded at 0.5
GEOMETRY = DATA_DIR / "geometries/ctscans/B_HAI_256.pt"
THRESHOLD = 0.5

# physics, clamped on the left edge, traction on the right edge
E_MIN, E_MAX = MATERIAL["E"]
NU = MATERIAL["nu"]  # plane strain
TRACTION = 1.0

# discretization
REFERENCE_DEGREE = 5  # about 2 % energy error, 0.05 % displacement error against p = 8
CG_TOL = 1e-10

# model settings
FEATURES = 64  # random Fourier features
SIGMA = 4.0
LAYERS = [2 * FEATURES, 128, 128, 128, 128, 2]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]

# ------------------------------------- load data -------------------------------------
g = torch.load(GEOMETRY, weights_only=False, map_location="cpu")[0].numpy() / 255.0
E = np.where(g >= THRESHOLD, E_MAX, E_MIN)  # E[i, j] of the pixel at (x_i, y_j)
R = E.shape[0]


# --------------------------------------- helper --------------------------------------
def fem(degree):
    """Q_p elements, one per voxel, preintegrated exactly.

    Returns the displacements at the voxel nodes, the potential energy and the time of
    the second solve (the first compiles the kernels).
    """
    nodes = [R * degree + 1] * 2
    fixed = np.zeros((2, *nodes), dtype=bool)
    fixed[:, 0, :] = True  # clamped left edge
    # plane strain as plane stress with E / (1 - nu^2) and nu / (1 - nu)
    solver = VoxelMultigrid(
        [R, R], [1.0, 1.0], "elasticity", fixed, degree, 1, NU / (1 - NU), floor=0.0
    )
    coeff = cp.asarray(E / (1 - NU**2), dtype=cp.float32)
    force = solver.face_load(0, 1, [TRACTION, 0.0])
    solver.update(coeff)
    solver.solve(force, rtol=CG_TOL)
    cp.cuda.Device().synchronize()
    tic = time.time()
    solver.update(coeff)
    u, _ = solver.solve(force, rtol=CG_TOL)
    cp.cuda.Device().synchronize()
    toc = time.time()
    u_nodes = cp.asnumpy(solver.voxel_nodes(u)).transpose(1, 2, 0)
    return u_nodes, -0.5 * float(force @ u), toc - tic


def density(x):
    """strain energy density, plane strain with the Lame constants of the pixel."""
    pixel = torch.clamp((x.detach() * R).long(), 0, R - 1)
    i, j = pixel[:, 0], pixel[:, 1]
    epsilon = strain(u_hat(x), x)
    trace = epsilon[:, 0] + epsilon[:, 1]
    squared = epsilon[:, 0] ** 2 + epsilon[:, 1] ** 2 + 0.5 * epsilon[:, 2] ** 2
    return (0.5 * lam[i, j] * trace**2 + mu[i, j] * squared)[:, None]


def potential(order):
    """potential energy of the network on a finer Gauss grid, in chunks."""
    x, w = gauss_grid([0.0, 0.0], [1.0, 1.0], R, order)
    x_right, w_right, _ = face([0.0, 0.0], [1.0, 1.0], R * order, 0, 1)
    u_right = u_hat(x_right.to(device))[:, 0:1]
    energy = -TRACTION * torch.sum(w_right.to(device) * u_right).item()
    for x_chunk, w_chunk in zip(x.split(2**16), w.split(2**16)):
        x_chunk = x_chunk.detach().to(device).requires_grad_()
        energy += torch.sum(w_chunk.to(device) * density(x_chunk)).item()
    return energy


def voxel_strain(u):
    """strains [eps_xx, eps_yy, eps_xy] at the voxel centres from the displacements at the voxel
    nodes, the gradient of the bilinear interpolant at the centre."""
    dx = 0.5 * (u[1:, :-1] - u[:-1, :-1] + u[1:, 1:] - u[:-1, 1:]) * R
    dy = 0.5 * (u[:-1, 1:] - u[:-1, :-1] + u[1:, 1:] - u[1:, :-1]) * R
    return np.stack([dx[..., 0], dy[..., 1], 0.5 * (dy[..., 0] + dx[..., 1])], -1)


def tensor_norm(epsilon):
    """pointwise Frobenius norm of the symmetric strain tensor."""
    return np.sqrt(epsilon[..., 0] ** 2 + epsilon[..., 1] ** 2 + 2 * epsilon[..., 2] ** 2)


def errors(u, energy):
    """relative L2 and Linf errors of the displacement (both components) and the strain (all
    components), and the relative energy error."""
    u_error, u_norm = np.linalg.norm(u - u_ref, axis=-1), np.linalg.norm(u_ref, axis=-1)
    epsilon_ref = voxel_strain(u_ref)
    strain_error, strain_norm = tensor_norm(voxel_strain(u) - epsilon_ref), tensor_norm(epsilon_ref)
    return (
        np.linalg.norm(u_error) / np.linalg.norm(u_norm),
        u_error.max() / u_norm.max(),
        np.linalg.norm(strain_error) / np.linalg.norm(strain_norm),
        strain_error.max() / strain_norm.max(),
        math.sqrt(max(energy - energy_ref, 0.0) / abs(energy_ref)),
    )


# ---------------------------------------- solve --------------------------------------
u_ref, energy_ref, time_ref = fem(REFERENCE_DEGREE)
u_fem, energy_fem, time_fem = fem(1)
cp.get_default_memory_pool().free_all_blocks()  # leave the GPU memory to the network

# ------------------------------------ prepare data -----------------------------------
lam = torch.tensor(E * NU / ((1 + NU) * (1 - 2 * NU)), dtype=torch.float32, device=device)
mu = torch.tensor(0.5 * E / (1 + NU), dtype=torch.float32, device=device)

x, w = gauss_grid([0.0, 0.0], [1.0, 1.0], R, 1)  # one point per voxel
x, w = x.detach().to(device).requires_grad_(), w.to(device)
x_right, w_right, _ = face([0.0, 0.0], [1.0, 1.0], R, 0, 1)
x_right, w_right = x_right.to(device), w_right.to(device)

# --------------------------- instantiate model & optimizer ---------------------------
B = (torch.randn(2, FEATURES) * SIGMA).to(device)
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
init_weights(model, ACTIVATIONS[0])

embed = lambda x: torch.cat([torch.sin(2 * math.pi * x @ B), torch.cos(2 * math.pi * x @ B)], 1)
u_hat = lambda x: x[:, 0:1] * model(embed(x))  # clamped, u = 0 at x = 0
cost_fun = lambda: energy_cost(
    [
        (x, w, density),
        (x_right, w_right, lambda x: -TRACTION * u_hat(x)[:, 0:1]),
    ]
)

# -------------------------------------- training -------------------------------------
torch.cuda.synchronize()
tic = time.time()
cost_history = train(cost_fun, model.parameters(), EPOCHS, LR, decay=DECAY)
torch.cuda.synchronize()
time_dem = time.time() - tic

# ----------------------------------- postprocessing ----------------------------------
nodes = torch.linspace(0.0, 1.0, R + 1, device=device)
x_nodes = torch.stack(torch.meshgrid(nodes, nodes, indexing="ij"), dim=-1).reshape(-1, 2)
with torch.no_grad():
    u_dem = u_hat(x_nodes).cpu().numpy().reshape(R + 1, R + 1, 2)
error_dem = errors(u_dem, potential(4))

rows = {"fem": (time_fem, *errors(u_fem, energy_fem)), "dem": (time_dem, *error_dem)}
print(f"reference p={REFERENCE_DEGREE}: time {time_ref:.2f} s")
for name, (elapsed, u_l2, u_inf, strain_l2, strain_inf, energy) in rows.items():
    print(
        f"{name}: time {elapsed:.3f} s, displacement {u_l2:.2e} (L2) {u_inf:.2e} (Linf), "
        f"strain {strain_l2:.2e} (L2) {strain_inf:.2e} (Linf), energy {energy:.2e}"
    )

fields = {"ref": u_ref, "fem": u_fem, "dem": u_dem}
epsilons = {name: voxel_strain(u) for name, u in fields.items()}
u_errors = {name: np.linalg.norm(fields[name] - u_ref, axis=-1) for name in rows}
strain_errors = {name: tensor_norm(epsilons[name] - epsilons["ref"]) for name in rows}
u_max = max(e.max() for e in u_errors.values())
strain_max = max(np.percentile(e, 99.9) for e in strain_errors.values())  # ignore single voxels
strain_min, strain_top = np.percentile(epsilons["ref"][..., 0], [0.5, 99.5])
figures = []
for name in fields:
    panels = [
        ("uy", fields[name][..., 1], {"cmap": "turbo", "vmin": u_ref[..., 1].min(), "vmax": u_ref[..., 1].max()}),
        ("exx", epsilons[name][..., 0], {"cmap": rainbow, "vmin": strain_min, "vmax": strain_top}),
    ]
    if name in rows:  # clipped at the lower limit, the error vanishes on the clamp
        panels += [
            ("uerror", np.maximum(u_errors[name], 1e-2 * u_max), {"cmap": "hot_r", "norm": LogNorm(1e-2 * u_max, u_max)}),
            ("exxerror", np.maximum(strain_errors[name], 1e-2 * strain_max), {"cmap": "hot_r", "norm": LogNorm(1e-2 * strain_max, strain_max)}),
        ]
    for key, field, style in panels:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), **style)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figures.append((fig, f"elasticity2D_benchmark_{name}_{key}.png"))

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(np.array(cost_history) - energy_ref, "k")
    ax.set_yscale("log")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for fig, name in figures:
        fig.savefig(RGB_PDF_DIR / name, dpi=128)
    print(f"pointwise displacement error from {1e-2 * u_max:.2e} to {u_max:.2e}")
    print(f"pointwise strain error from {1e-2 * strain_max:.2e} to {strain_max:.2e}")
    save_csv(
        CSV_DIR / "elasticity2D_benchmark_errors.csv",
        method=list(rows),
        time=[row[0] for row in rows.values()],
        displacement=[row[1] for row in rows.values()],
        displacement_inf=[row[2] for row in rows.values()],
        strain=[row[3] for row in rows.values()],
        strain_inf=[row[4] for row in rows.values()],
        energy=[row[5] for row in rows.values()],
    )
    save_csv(
        CSV_DIR / "elasticity2D_benchmark_dem_cost_history.csv",
        cost=np.array(cost_history),
    )
