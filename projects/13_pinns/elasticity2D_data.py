import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch
from torch import nn

from DL import init_weights
from NN import MLP
from helper import (
    energy_cost,
    energy_error,
    in_holes,
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
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 5000
LR = 1e-3
RESOLUTION = 50
TEST_RESOLUTION = 640
SENSORS = [0, 4, 16, 64, 256, 1024]  # displacement measurements from the reference
DATA_WEIGHT = 100.0  # penalty on the squared misfit, for the network and the elements

# physics
E = 1.0
NU = 0.3
TRACTION = 1.0
HOLES = 1

# model settings
LAYERS = [2, 64, 64, 64, 2]

# finite element comparison
NELEMENTS = 5  # the finite element error without data matches the network
DEGREE = 2


# --------------------------------------- helper --------------------------------------
def assemble():
    centers = (np.arange(NELEMENTS) + 0.5) / NELEMENTS
    centers = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=-1)
    base_grid = mlhp.makeGrid([NELEMENTS] * 2, [1.0] * 2, [0.0] * 2)
    mask = (~in_holes(centers, HOLES)).ravel("C").tolist()
    grid = mlhp.makeRefinedGrid(mlhp.makeFilteredGrid(base_grid, mask=mask))
    basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=2)

    zero = mlhp.vectorField(2, [0.0, 0.0])
    dirichlet = mlhp.integrateDirichletDofs(zero, basis, [0])
    matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)
    constitutive = mlhp.planeStressMaterial(
        mlhp.scalarField(2, E), mlhp.scalarField(2, NU)
    )
    integrand = mlhp.staticDomainIntegrand(
        mlhp.smallStrainKinematics(2), constitutive, zero
    )
    mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)
    traction = mlhp.neumannIntegrand(mlhp.vectorField(2, [TRACTION, 0.0]))
    right = mlhp.quadratureOnMeshFaces(grid, [1])
    mlhp.integrateOnSurface(basis, traction, [vector], right, dirichletDofs=dirichlet)

    K = sp.csr_matrix(tuple(np.array(a) for a in matrix.csr_arrays[0]), matrix.shape)
    interior = np.setdiff1d(np.arange(basis.ndof()), dirichlet[0])
    return basis, K, vector.array, interior


def relative_error(u):
    return np.sqrt(np.nansum((u - u_all) ** 2) / np.nansum(u_all**2))


def shape_functions(basis, interior, x_sensors):
    coordinates = [x_sensors[:, 0].tolist(), x_sensors[:, 1].tolist()]
    columns = []
    for dof in interior:
        unit = np.zeros(basis.ndof())
        unit[dof] = 1.0
        values = mlhp.vectorEvaluator(basis, mlhp.DoubleVector(unit.tolist()))
        columns.append(np.array(values(*coordinates)).reshape(-1))
    return np.stack(columns, axis=1)


# -------------------------------------- data study -----------------------------------
reference = np.load(DATA_DIR / "elasticity2D_reference.npz")
compliance = reference["compliance"][list(reference["holes"]).index(HOLES)]
x_all = np.stack([reference["x"], reference["y"]], axis=-1).reshape(-1, 2)
u_all = np.stack([reference[f"ux_{HOLES}"], reference[f"uy_{HOLES}"]], -1)
u_all = u_all.reshape(-1, 2)
order = rng.permutation(np.flatnonzero(~np.isnan(u_all[:, 0])))

C = plane_stress(E, NU).to(device)
t_right = torch.tensor([[TRACTION, 0.0]], device=device)
x, w, x_right, w_right = plate_points(HOLES, RESOLUTION)
x, w = x.detach().to(device).requires_grad_(), w.to(device)
x_right, w_right = x_right.detach().to(device), w_right.to(device)

basis, K, f, interior = assemble()
N_all = shape_functions(basis, interior, x_all[order[: max(SENSORS)]])

results = {"sensors": SENSORS}
for key in ["dem", "dem_l2", "dem_time", "fem", "fem_l2", "fem_time"]:
    results[key] = []
for sensors in SENSORS:
    x_sensors = torch.tensor(x_all[order[:sensors]], dtype=torch.float32, device=device)
    u_sensors = torch.tensor(u_all[order[:sensors]], dtype=torch.float32, device=device)

    activations = [nn.Tanh() for _ in range(len(LAYERS) - 2)]
    model = MLP(LAYERS, activations)
    model.to(device)
    init_weights(model, activations[0])
    u_hat = lambda x: x[:, 0:1] * model(x)

    def density(x):
        epsilon = strain(u_hat(x), x)
        return 0.5 * torch.sum(epsilon * (epsilon @ C), 1, keepdim=True)

    cost_fun = lambda: (
        energy_cost(
            [(x, w, density), (x_right, w_right, lambda x: -u_hat(x) @ t_right.T)]
        )
        + 0.5 * DATA_WEIGHT * torch.sum((u_hat(x_sensors) - u_sensors) ** 2)
    )

    tic = time.time()
    train(cost_fun, model.parameters(), EPOCHS, LR)
    toc = time.time()
    potential = plate_potential(u_hat, C, t_right, HOLES, TEST_RESOLUTION, device)
    results["dem"].append(energy_error(potential, compliance))
    x_test = torch.tensor(x_all, dtype=torch.float32, device=device)
    results["dem_l2"].append(relative_error(u_hat(x_test).detach().cpu().numpy()))
    results["dem_time"].append(toc - tic)

    tic = time.time()
    N = N_all[: 2 * sensors]
    u_measured = u_all[order[:sensors]].reshape(-1)
    A = K + DATA_WEIGHT * sp.csr_matrix(N.T @ N)
    b = f + DATA_WEIGHT * N.T @ u_measured
    u_fem = spl.spsolve(A.tocsc(), b)
    toc = time.time()
    potential = 0.5 * u_fem @ (K @ u_fem) - f @ u_fem
    results["fem"].append(energy_error(potential, compliance))
    dofs = np.zeros(basis.ndof())
    dofs[interior] = u_fem
    displacement = mlhp.vectorEvaluator(basis, mlhp.DoubleVector(dofs.tolist()))
    u_fem = np.array(displacement(*x_all.T.tolist())).reshape(-1, 2)
    results["fem_l2"].append(relative_error(u_fem))
    results["fem_time"].append(toc - tic)
    print(
        f"{sensors} sensors, energy / displacement error "
        f"network {results['dem'][-1]:.2e} / {results['dem_l2'][-1]:.2e}, "
        f"finite elements {results['fem'][-1]:.2e} / {results['fem_l2'][-1]:.2e}"
    )
results = {key: np.array(value) for key, value in results.items()}

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, suffix in zip(axs, ["", "_l2"]):
        ax.semilogy(results["sensors"], results[f"dem{suffix}"], "r.-")
        ax.semilogy(results["sensors"], results[f"fem{suffix}"], "b.--")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("sensors")
    axs[0].set_ylabel("relative energy error")
    axs[1].set_ylabel("relative displacement error")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "elasticity2D_data.csv", **results)
