import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import torch
from torch import nn

from DL import differentiate, init_weights
from NN import MLP
from helper import (
    corner_plot_points,
    energy_cost,
    graded_quadrature,
    pinn_cost,
    sample,
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
DIMENSIONS = [1, 2, 3, 4, 5, 6]

# hyperparameters
EPOCHS = 10000
LR = 1e-3
SAMPLES = 2**14  # quadrature points in the domain
BOUNDARY_SAMPLES = 2**10  # quadrature points per dirichlet face
ERROR_LEVELS = 20  # graded cells towards the corner for the energy error, as in mlhp
ERROR_ORDER = 3  # gauss points per direction and cell
PENALTY = 100.0  # dirichlet penalty weight
EVALUATE_EVERY = 100
PLOT_DIMENSIONS = [1, 2, 3]
PLOT_RESOLUTION = 100  # as in poisson_corner_reference.py

# physics
GAMMA = 0.65  # 1D only, exponent of x**GAMMA - GAMMA * x

# model settings
HIDDEN_LAYERS = [64, 64, 64]


# --------------------------------------- helper --------------------------------------
def u_fun(x):
    if x.shape[1] == 1:
        return x**GAMMA - GAMMA * x
    return torch.linalg.norm(x, dim=1, keepdim=True) ** 0.5


def dirichlet_points(D):
    if D == 1:
        return torch.zeros(1, 1)
    points = []
    for axis in range(D):
        x_face, _ = sample([0.0] * (D - 1), [1.0] * (D - 1), BOUNDARY_SAMPLES, "sobol")
        x_face = x_face.detach()
        ones = torch.ones(len(x_face), 1)
        points.append(torch.cat([x_face[:, :axis], ones, x_face[:, axis:]], dim=1))
    return torch.cat(points)


def plot_field(field, D, norm, cmap):
    fig = plt.figure(figsize=(4, 4))
    if D == 2:
        ax = fig.add_subplot()
        field = field.reshape(PLOT_RESOLUTION, PLOT_RESOLUTION)
        mesh = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap)
        mesh.set_norm(norm)
        ax.axis("off")
    else:
        ax = fig.add_subplot(projection="3d")
        a = (np.arange(PLOT_RESOLUTION) + 0.5) / PLOT_RESOLUTION
        a, b = np.meshgrid(a, a, indexing="ij")
        one = np.ones_like(a)
        # mirrored so that the singular corner points towards the viewer
        faces = [(one, 1 - a, 1 - b), (1 - a, one, 1 - b), (1 - a, 1 - b, one)]
        for i, (X, Y, Z) in enumerate(faces):
            values = field.reshape(3, PLOT_RESOLUTION, PLOT_RESOLUTION)[i]
            colors = plt.get_cmap(cmap)(norm(values))
            ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=colors,
                shade=False,
                rasterized=True,
                rcount=PLOT_RESOLUTION,
                ccount=PLOT_RESOLUTION,
            )
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=45)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------ dimension study --------------------------------
reference = np.load(DATA_DIR / "poisson_corner_reference.npz")
histories = {}
plots = {}
for D in DIMENSIONS:
    x, w = sample([0.0] * D, [1.0] * D, SAMPLES, "sobol")
    x = x.detach().to(device).requires_grad_()
    w = w.to(device)
    x_dirichlet = dirichlet_points(D).to(device)
    x_test, w_test = graded_quadrature(D, ERROR_LEVELS, ERROR_ORDER)
    x_test = x_test.to(device).requires_grad_()
    w_test = w_test.to(device)

    grad_u = lambda x: differentiate(u_fun(x), x)
    laplacian = lambda x: sum(
        differentiate(grad_u(x)[:, i : i + 1], x)[:, i : i + 1] for i in range(D)
    )
    f = -laplacian(x).detach()
    g = u_fun(x_dirichlet)
    grad_u_test = torch.cat([grad_u(x).detach() for x in x_test.split(2**16)])

    model = MLP([D, *HIDDEN_LAYERS, 1], [nn.Tanh() for _ in HIDDEN_LAYERS])
    model.to(device)
    init_weights(model, nn.Tanh())

    u_hat = lambda x: x * model(x) if D == 1 else model(x)  # 1D source not integrable
    grad_u_hat = lambda x: differentiate(u_hat(x), x)
    cost_fun = lambda: (
        energy_cost(
            [
                (
                    x,
                    w,
                    lambda x: (
                        0.5 * torch.sum(grad_u_hat(x) ** 2, 1, keepdim=True)
                        - f * u_hat(x)
                    ),
                )
            ]
        )
        + PENALTY * pinn_cost([u_hat(x_dirichlet) - g])
    )

    history = {"time": [], "error": []}
    clock = {"training": 0.0, "last": time.time()}

    def evaluate(epoch):
        if epoch % EVALUATE_EVERY != 0 and epoch != EPOCHS - 1:
            return
        synchronize()
        clock["training"] += time.time() - clock["last"]
        grad_u_hat_test = [grad_u_hat(x).detach() for x in x_test.split(2**16)]
        difference = torch.cat(grad_u_hat_test) - grad_u_test
        error = torch.sum(w_test * difference**2) / torch.sum(w_test * grad_u_test**2)
        error = torch.sqrt(error)
        history["time"].append(clock["training"])
        history["error"].append(error.item())
        clock["last"] = time.time()

    train(cost_fun, model.parameters(), EPOCHS, LR, callback=evaluate)
    histories[D] = {key: np.array(value) for key, value in history.items()}
    if D in PLOT_DIMENSIONS:
        x_plot = corner_plot_points(D, PLOT_RESOLUTION)
        x_plot = torch.tensor(x_plot, dtype=torch.float32, device=device)
        plots[D] = {
            "u": u_fun(x_plot).cpu().numpy().ravel(),
            "dem": u_hat(x_plot).detach().cpu().numpy().ravel(),
            "fem": reference[f"u_{D}D"],
        }

# ----------------------------------- postprocessing ----------------------------------
scores = {"dim": np.array(DIMENSIONS), "fem": [], "dem": []}
print("dim | mlhp ndof  error     time     | network error  time")
for D in DIMENSIONS:
    mask = reference["dim"] == D
    ndof, error, elapsed = [
        reference[key][mask][-1] for key in ["ndof", "error", "time"]
    ]
    network = histories[D]
    print(
        f"{D}D  | {ndof:9d} {error:.2e} {elapsed:8.2f} s | "
        f"{network['error'][-1]:.2e} {network['time'][-1]:8.2f} s"
    )
    within_budget = reference["time"][mask] <= network["time"][-1]
    scores["fem"].append(np.min(reference["error"][mask][within_budget]))
    scores["dem"].append(network["error"][-1])
scores = {key: np.array(value) for key, value in scores.items()}
scores["ratio"] = scores["fem"] / scores["dem"]

figures = []
for D, fields in plots.items():
    if D == 1:
        continue
    errors = {key: np.abs(fields[key] - fields["u"]) for key in ["fem", "dem"]}
    norm = plt.Normalize(np.min(fields["u"]), np.max(fields["u"]))
    figures.append((plot_field(fields["u"], D, norm, "cividis"), f"{D}D_u"))
    positive = np.concatenate(list(errors.values()))
    positive = positive[positive > 0]
    norm = LogNorm(np.percentile(positive, 1), np.max(positive))
    for key, error in errors.items():
        figures.append((plot_field(error, D, norm, "hot_r"), f"{D}D_{key}"))

if not args.book:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for D in DIMENSIONS:
        mask = reference["dim"] == D
        axs[0].loglog(reference["time"][mask], reference["error"][mask], "b.-")
        axs[0].loglog(histories[D]["time"][1:], histories[D]["error"][1:], "r")
    axs[0].set_xlabel("time in s")
    axs[0].set_ylabel("relative energy error")
    axs[1].semilogy(scores["dim"], scores["ratio"], "k.-")
    axs[1].set_xlabel("dimension")
    axs[1].set_ylabel("finite element error / network error at equal time")

    fig, axs = plt.subplots(1, 2)
    x_plot = corner_plot_points(1, PLOT_RESOLUTION).ravel()
    axs[0].semilogx(x_plot, plots[1]["u"], "k")
    for key, style in [("fem", "b"), ("dem", "r")]:
        axs[1].loglog(x_plot, np.abs(plots[1][key] - plots[1]["u"]), style)
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for D in DIMENSIONS:
        mask = reference["dim"] == D
        save_csv(CSV_DIR / f"poisson_corner_{D}D.csv", **histories[D])
        save_csv(
            CSV_DIR / f"poisson_corner_reference_{D}D.csv",
            ndof=reference["ndof"][mask],
            time=reference["time"][mask],
            error=reference["error"][mask],
        )
    save_csv(CSV_DIR / "poisson_corner_scores.csv", **scores)
    save_csv(
        CSV_DIR / "poisson_corner_fields_1D.csv",
        x=corner_plot_points(1, PLOT_RESOLUTION).ravel(),
        u=plots[1]["u"],
        fem=np.abs(plots[1]["fem"] - plots[1]["u"]),
        dem=np.abs(plots[1]["dem"] - plots[1]["u"]),
    )
    for fig, name in figures:
        fig.savefig(RGB_PDF_DIR / f"poisson_corner_{name}.pdf")
