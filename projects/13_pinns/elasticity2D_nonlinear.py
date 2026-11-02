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
from helper import energy_cost, face, gauss_grid, train
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
# nonlinearity, indices into the tractions of the reference, one shows its fields
PROBLEMS = {  # direction of the traction on the right edge
    "cantilever": [0.0, -1.0],
    "column": [-1.0, -1e-3],  # axial compression with a lateral imperfection
}
LOADS = [6]
if args.book:  # the study of the book figure, all tractions of the reference
    LOADS = None
FIELDS = [1e-3, 1e-1, 1e1]  # target nonlinearities of the fields in the book

# hyperparameters
EPOCHS = 3000  # after the extreme learning machine initialization
LR = 1e-3
DECAY = 0.9998

# quadrature, Gauss points on a uniform grid
RESOLUTION = [400, 40]  # cells along and across the beam
ORDER = 2  # Gauss points per direction and cell

# physics, plane strain neo-hookean beam, clamped on the left; its strains stay small, the nonlinearity is geometric
LENGTHS = [1.0, 0.1]
E = 1.0
NU = 0.3
COMPLIANCE = 4 * LENGTHS[0] ** 3 * (1 - NU**2) / (E * LENGTHS[1] ** 2)  # tip deflection per traction, beam theory
LAM, MU = E * NU / ((1 + NU) * (1 - 2 * NU)), 0.5 * E / (1 + NU)

# model settings
FEATURES = 64  # random Fourier features
SIGMA = 1.0  # best of 0.5, 1, 2 (and learning rates 1e-3, 1e-4) at the smallest and largest load
LAYERS = [2 * FEATURES, 128, 128, 128, 128, 2]


# --------------------------------------- helper --------------------------------------
def neo_hookean(x):
    u = u_hat(x)
    du = [torch.autograd.grad(u[:, i].sum(), x, create_graph=True)[0] for i in range(2)]
    F11, F12, F21, F22 = 1 + du[0][:, 0], du[0][:, 1], du[1][:, 0], 1 + du[1][:, 1]
    log_J = torch.log(torch.clamp(F11 * F22 - F12 * F21, min=1e-6))  # inverted, no energy
    trace_C = F11**2 + F12**2 + F21**2 + F22**2
    return (0.5 * MU * (trace_C - 2 - 2 * log_J) + 0.5 * LAM * log_J**2)[:, None]


def last_layer_basis(x):
    """g = x_0 [h, 1] with the hidden features h of the network and its x and y
    derivatives, u = t g W for the last layer W (with its bias)."""
    ones, zeros = torch.ones(x.shape[0], 1, device=device), torch.zeros(x.shape[0], 1, device=device)
    hidden = lambda x: model.model[:-1](embed(x))
    h, h_x = jvp(hidden, (x,), (torch.cat([ones, zeros], 1),))
    _, h_y = jvp(hidden, (x,), (torch.cat([zeros, ones], 1),))
    h, h_x, h_y = torch.cat([h, ones], 1), torch.cat([h_x, zeros], 1), torch.cat([h_y, zeros], 1)
    return x[:, 0:1] * h, h + x[:, 0:1] * h_x, x[:, 0:1] * h_y


def initialize_last_layer(chunk=8192):
    """extreme learning machine on the linear-elastic energy, quadratic in the last
    layer, so its minimizer per unit traction solves K w = f / COMPLIANCE."""
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
    K_elm = torch.cat([torch.cat(blocks[:2], 1), torch.cat([blocks[1].T, blocks[2]], 1)])
    with torch.no_grad():
        g, _, _ = last_layer_basis(x_right)
    f = (w_right.reshape(-1, 1) * g).sum(0).double()
    f = torch.cat([PROBLEMS[problem][0] * f, PROBLEMS[problem][1] * f]) / COMPLIANCE
    # minimum-norm solution, the random features are nearly linearly dependent
    eigenvalues, V = torch.linalg.eigh(K_elm)
    keep = eigenvalues > 1e-12 * eigenvalues.max()
    solution = (V[:, keep] @ ((V[:, keep].T @ f) / eigenvalues[keep])).float().reshape(2, -1)
    with torch.no_grad():
        model.model[-1].weight.copy_(solution[:, :-1])
        model.model[-1].bias.copy_(solution[:, -1])


def mirror(u):
    """mirror image about the beam axis, the other buckling direction."""
    return np.stack([u[:, ::-1, 0], -u[:, ::-1, 1]], axis=-1)


# ------------------------------------- load data -------------------------------------
x, w = gauss_grid([0.0, 0.0], LENGTHS, RESOLUTION, ORDER)
x, w = x.detach().to(device).requires_grad_(), w.to(device)
x_right, w_right, _ = face([0.0, 0.0], LENGTHS, [2 * n for n in RESOLUTION], 0, 1)
x_right, w_right = x_right.detach().to(device), w_right.to(device)

# -------------------------------- nonlinearity study ---------------------------------
references, results = {}, {}
for problem in PROBLEMS:
    reference = np.load(DATA_DIR / f"elasticity2D_nonlinear_{problem}_reference.npz")
    x_test = np.stack([reference["x"], reference["y"]], axis=-1).reshape(-1, 2)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    references[problem] = reference
    results[problem] = {"nonlinearity": [], "error": [], "time": []}

    for load in LOADS or range(len(reference["traction"])):
        traction = float(reference["traction"][load])
        t = torch.tensor([[traction * d for d in PROBLEMS[problem]]], device=device)
        torch.manual_seed(0)
        B = (torch.randn(2, FEATURES) * SIGMA).to(device)
        activations = [nn.Tanh() for _ in range(len(LAYERS) - 2)]
        model = MLP(LAYERS, activations)
        model.to(device)
        init_weights(model, activations[0])
        embed = lambda x: torch.cat([torch.sin(2 * math.pi * x @ B), torch.cos(2 * math.pi * x @ B)], 1)
        u_hat = lambda x: traction * COMPLIANCE * x[:, 0:1] * model(embed(x))  # clamped, in linear tip deflections
        # per linear energy, else gradients at small loads fall below the epsilon of adam
        scale = traction**2 * COMPLIANCE * LENGTHS[1]
        cost_fun = lambda: energy_cost([(x, w, neo_hookean), (x_right, w_right, lambda x: -u_hat(x) @ t.T)]) / scale

        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.time()
        initialize_last_layer()
        train(cost_fun, model.parameters(), EPOCHS, LR, decay=DECAY)
        if device.type == "cuda":
            torch.cuda.synchronize()
        toc = time.time()

        u_ref = np.stack([reference[f"ux_{traction}"], reference[f"uy_{traction}"]], -1)
        with torch.no_grad():
            u = u_hat(x_test).cpu().numpy().reshape(u_ref.shape)
        # the column may buckle to either side, the error is against the nearer one
        errors = [np.linalg.norm(v - u_ref) / np.linalg.norm(u_ref) for v in [u, mirror(u)]]
        error, nonlinearity = min(errors), reference["nonlinearity"][load]
        print(
            f"{problem} traction {traction:.3e}, nonlinearity {nonlinearity:.2e}: "
            f"time {toc - tic:.1f} s, displacement error {error:.2e} ({'mirrored' if errors[1] < errors[0] else 'same side'}), "
            f"finite elements p=1 {reference['p1_error'][load]:.2e}, p=2 {reference['p2_error'][load]:.2e}"
        )
        for key, value in zip(["nonlinearity", "error", "time"], [nonlinearity, error, toc - tic]):
            results[problem][key].append(value)

# ----------------------------------- postprocessing ----------------------------------
figures, tips = [], {"problem": [], "nonlinearity": [], "x": [], "y": []}
for problem, reference in references.items():
    for target in FIELDS:
        load = int(np.argmin(np.abs(np.log(reference["target"] / target))))
        traction = float(reference["traction"][load])
        u = np.stack([reference[f"ux_{traction}"], reference[f"uy_{traction}"]], -1)
        magnitude = np.linalg.norm(u, axis=-1)
        fig, ax = plt.subplots(figsize=(4, 4 * 1.25 / 1.15))
        mesh = ax.pcolormesh(reference["x"] + u[..., 0], reference["y"] + u[..., 1], magnitude, cmap="turbo", shading="gouraud")
        mesh.set_rasterized(True)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-1.1, 0.15)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figures.append((fig, f"elasticity2D_nonlinear_{problem}_{target:.0e}.png"))
        middle = u.shape[1] // 2  # deformed position of the loaded edge, for the load arrow
        for key, value in zip(tips, [problem, reference["nonlinearity"][load], 1 + u[-1, middle, 0], LENGTHS[1] / 2 + u[-1, middle, 1]]):
            tips[key].append(value)

if not args.book:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for problem, reference in references.items():
        axs[0].loglog(results[problem]["nonlinearity"], results[problem]["error"], "r.")
        axs[1].loglog(results[problem]["nonlinearity"], results[problem]["time"], "r.")
        for degree, style in [(1, "b.--"), (2, "k.--")]:
            axs[0].loglog(reference["nonlinearity"], reference[f"p{degree}_error"], style)
            axs[1].loglog(reference["nonlinearity"], reference[f"p{degree}_time"], style)
    axs[0].set_ylabel("relative displacement error")
    axs[1].set_ylabel("time in s")
    for ax in axs:
        ax.set_xlabel("nonlinearity")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for fig, name in figures:
        fig.savefig(RGB_PDF_DIR / name, dpi=128, transparent=True)
    save_csv(CSV_DIR / "elasticity2D_nonlinear_fields.csv", **tips)
    for problem, reference in references.items():
        save_csv(CSV_DIR / f"elasticity2D_nonlinear_{problem}_dem.csv", **results[problem])
        for degree in [1, 2]:
            save_csv(
                CSV_DIR / f"elasticity2D_nonlinear_{problem}_fem_p{degree}.csv",
                nonlinearity=reference["nonlinearity"],
                traction=reference["traction"],
                error=reference[f"p{degree}_error"],
                ndof=reference[f"p{degree}_ndof"],
                iterations=reference[f"p{degree}_iterations"],
                time=reference[f"p{degree}_time"],
            )
