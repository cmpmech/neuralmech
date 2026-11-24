import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# TARGET = "gaussian"  # gaussian (recover a known beam) or meltpool (uniform melt pool)
TARGET = "meltpool"
# geometry (micrometers, scan along x, depth along y)
LENGTHS = [1400.0, 300.0]
CENTER = 450.0  # laser position on the top edge
MELTPOOL = [330.0, 570.0, 30.0]  # x from, x to, depth of the pinned half ellipse inside the pool
VIEW = [200.0, 800.0, 100.0, 300.0]  # x and y range of the book images

# physics (micrometers, microseconds, degrees celsius, stainless steel 316L)
VELOCITY = 0.8  # 800 mm/s scan speed, material moves in +x relative to the laser
CONDUCTIVITY = 25.0  # W/(m C), constant at a mean temperature
DENSITY = 7984.0  # kg/m^3
HEAT_CAPACITY = 550.0  # J/(kg C)
ABSORPTIVITY = 0.3
T0 = 20.0
T_MELT = 1390.0  # liquidus
T_POOL = 1700.0  # pinned core temperature, the melt pool forms around the core
T_CAP = 2500.0  # penalized above, keeps the pool below boiling
CAP_WEIGHT = 10.0

# gaussian beam of the verification target
PEAK = 1.0  # MW/cm^2
SIGMA = 50.0  # D4sigma diameter of 200 um

# discretization
DEGREE = 1
NELEMENTS = [280, 60]
PENALTY = 1e3  # weight pinning the melt pool temperature when building the target

# optimization
CONTROLS = 51  # hat functions on the top edge, kinks on element edges
WINDOW = 250.0  # half width of the controlled part of the top edge
ITERS = 500

kappa = CONDUCTIVITY / (DENSITY * HEAT_CAPACITY) * 1e6  # um^2/us
flux_scale = ABSORPTIVITY * 1e10 / (DENSITY * HEAT_CAPACITY)  # MW/cm^2 -> C um/us
x_controls = np.linspace(CENTER - WINDOW, CENTER + WINDOW, CONTROLS)
spacing = x_controls[1] - x_controls[0]


# --------------------------------------- helper --------------------------------------
def to_scipy(matrix):
    data = np.asarray(matrix.data_array)
    indices = np.asarray(matrix.indices_array)
    indptr = np.asarray(matrix.indptr_array)
    return sp.csr_matrix((data, indices, indptr), shape=tuple(matrix.shape))


def assemble(integrand):
    matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)
    mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)
    return to_scipy(matrix), np.asarray(vector).copy()


def assemble_flux(expression):
    vector = mlhp.allocateRhsVector(mlhp.allocateSparseMatrix(basis, dirichlet[0]))
    flux = mlhp.vectorField(2, [mlhp.scalarField(2, f"{flux_scale} * ({expression})")])
    mlhp.integrateOnSurface(
        basis, mlhp.neumannIntegrand(flux), [vector], top, dirichletDofs=dirichlet
    )
    return np.asarray(vector).copy()


def loss_and_gradient(beta):
    T = lu.solve(f0 + B @ beta)
    residual = T - T_target
    excess = np.maximum(T - T_CAP, 0.0)
    lam = lu.solve(M_masked @ residual + CAP_WEIGHT * lumped * excess, trans="T")
    J = 0.5 * residual @ M_masked @ residual + 0.5 * CAP_WEIGHT * lumped @ excess**2
    return J / J0, B.T @ lam / J0


def temperature_field(T):
    dofs = mlhp.inflateDofs(mlhp.DoubleVector(T), dirichlet)
    result = mlhp.DataAccumulator()
    processors = [mlhp.solutionProcessor(2, dofs, "Temperature")]
    cellmesh = mlhp.gridCellMesh([DEGREE + 1] * 2)
    mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)
    return result.triangulation(mpl=True), np.array(result.data()[0])


# ---------------------------------------- setup --------------------------------------
grid = mlhp.makeRefinedGrid(NELEMENTS, LENGTHS)
basis = mlhp.makeHpTensorSpace(grid, degree=DEGREE, nfields=1)
top = mlhp.quadratureOnMeshFaces(grid, [3])

# inflow at T0, the laser flux enters through the top, the rest is insulated
dirichlet = mlhp.integrateDirichletDofs(mlhp.scalarField(2, T0), basis, [0])

velocity = mlhp.vectorField(2, [VELOCITY, 0.0])
advection = mlhp.advectionDiffusionIntegrand(
    velocity, mlhp.scalarField(2, kappa), mlhp.scalarField(2, 0.0)
)
K, f0 = assemble(advection)
mass = mlhp.l2DomainIntegrand(
    rhs=mlhp.scalarField(2, 0.0), mass=mlhp.scalarField(2, 1.0)
)
M, _ = assemble(mass)

hat = "(1 - abs(x - {0}) / {1}) if abs(x - {0}) < {1} else 0"
B = np.stack([assemble_flux(hat.format(xi, spacing)) for xi in x_controls], axis=1)

beam_true = PEAK * np.exp(-((x_controls - CENTER) ** 2) / (2 * SIGMA**2))
if TARGET == "gaussian":
    gaussian = f"{PEAK} * exp(-(x - {CENTER})**2 / (2 * {SIGMA}**2))"
    T_target = splu(K.tocsc()).solve(f0 + assemble_flux(gaussian))
else:
    x1, x2, depth = MELTPOOL
    ex = f"(2 * x - {x1} - {x2}) / ({x2} - {x1})"
    ey = f"(y - {LENGTHS[1]}) / {depth}"
    inside = f"min(1, max(0, 2 * (1 - ({ex}) * ({ex}) - ({ey}) * ({ey}))))"
    pin = mlhp.l2DomainIntegrand(
        rhs=mlhp.scalarField(2, f"{PENALTY * T_POOL} * {inside}"),
        mass=mlhp.scalarField(2, f"{PENALTY} * {inside}"),
    )
    K_pin, f_pin = assemble(pin)
    T_target = splu((K + K_pin).tocsc()).solve(f0 + f_pin)

# misfit only outside the target melt pool, the temperature inside is left free
T_target = np.maximum(T_target, T0)
outside = sp.diags((T_target < T_MELT) * 1.0)
M_masked = outside @ M @ outside
lumped = np.asarray(M.sum(axis=1)).ravel()

lu = splu(K.tocsc())
residual0 = lu.solve(f0) - T_target
J0 = 0.5 * residual0 @ M_masked @ residual0  # unheated plate, reported losses are relative

# ------------------------------------ optimization -----------------------------------
history = []
tic = time.time()
result = minimize(
    loss_and_gradient,
    np.zeros(CONTROLS),
    jac=True,
    method="L-BFGS-B",
    bounds=[(0.0, None)] * CONTROLS,
    callback=lambda intermediate_result: history.append(intermediate_result.fun),
    options={"maxiter": ITERS, "ftol": 1e-14, "gtol": 1e-12},
)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s for {result.nit} iter, loss {result.fun:.2e}")
beta = result.x

# ----------------------------------- postprocessing ----------------------------------
T_opt = lu.solve(f0 + B @ beta)
# each field keeps its own triangulation, the output cell order is thread dependent
fields = [temperature_field(T_target), temperature_field(T_opt)]
temperatures = np.concatenate([temperature for _, temperature in fields])
levels = np.linspace(temperatures.min(), temperatures.max(), 64)

figs = []
for tri, temperature in fields:
    fig = plt.figure(figsize=(3.5, 3.5 * (VIEW[3] - VIEW[2]) / (VIEW[1] - VIEW[0])), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.tricontourf(tri, temperature, levels=levels, cmap="inferno")
    ax.tricontour(tri, temperature, levels=[T_MELT], colors="w", linewidths=0.8)
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.axis("off")
    ax.set_rasterized(True)
    figs.append(fig)

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(x_controls, beta, "k")
    if TARGET == "gaussian":
        axes[0].plot(x_controls, beam_true, "b--")
    axes[0].set_xlabel("x [um]")
    axes[0].set_ylabel("intensity [MW/cm^2]")
    axes[1].semilogy(history, "k")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative loss")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for fig, name in zip(figs, ["target", "optimized"]):
        fig.savefig(RGB_PDF_DIR / f"beam_shape_{TARGET}_{name}.pdf")
    plt.close("all")
    beam_csv = CSV_DIR / f"beam_shape_{TARGET}_beam.csv"
    save_csv(beam_csv, x=x_controls, beta=beta, true=beam_true)
    save_csv(CSV_DIR / f"beam_shape_{TARGET}_loss.csv", loss=history)
