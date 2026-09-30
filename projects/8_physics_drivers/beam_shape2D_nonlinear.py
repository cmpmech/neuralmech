import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse as sp
from numba import carray, cfunc, types
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
MELTPOOL = [300.0, 600.0, 60.0]  # x from, x to, depth of the target melt pool

# physics (micrometers, microseconds, degrees celsius, stainless steel 316L)
VELOCITY = 0.8  # 800 mm/s scan speed, material moves in +x relative to the laser
CONDUCTIVITY = [13.6, 15.3e-3]  # k(T) = k0 + k1 T in W/(m C)
HEAT_CAPACITY = [472.0, 101.02e-3]  # cs(T) = cs0 + cs1 T in J/(kg C)
DENSITY = 7984.0  # kg/m^3
LATENT_HEAT = 2.8e5  # J/kg
T_SOLIDUS, T_LIQUIDUS = 1290.0, 1390.0
SMOOTHNESS = 1.6  # widens the regularized phase change beyond the melting range
ABSORPTIVITY = 0.3
T0 = 20.0
T_POOL = 1400.0  # target inside the melt pool, just above the liquidus

# gaussian beam of the verification target
PEAK = 1.0  # MW/cm^2
SIGMA = 50.0  # D4sigma diameter of 200 um

# discretization
DEGREE = 1
NELEMENTS = [280, 60]
PENALTY = 1e3  # weight pinning the melt pool temperature when building the target

# newton
NEWTON_ITERS = 30
NEWTON_TOL = 1e-10  # relative temperature update

# optimization
CONTROLS = 51  # hat functions on the top edge, kinks on element edges
WINDOW = 250.0  # half width of the controlled part of the top edge
ITERS = 200

# all heat capacities relative to rho cs0, so conductivities become um^2/us
reference = DENSITY * HEAT_CAPACITY[0]
parameters = [
    VELOCITY,
    HEAT_CAPACITY[1] / HEAT_CAPACITY[0],
    LATENT_HEAT / HEAT_CAPACITY[0],
    0.5 * (T_LIQUIDUS + T_SOLIDUS),
    0.5 * SMOOTHNESS * (T_LIQUIDUS - T_SOLIDUS),
    CONDUCTIVITY[0] / reference * 1e6,
    CONDUCTIVITY[1] / reference * 1e6,
]
flux_scale = ABSORPTIVITY * 1e10 / reference  # MW/cm^2 -> C um/us
x_controls = np.linspace(CENTER - WINDOW, CENTER + WINDOW, CONTROLS)
spacing = x_controls[1] - x_controls[0]


# --------------------------------------- helper --------------------------------------
@cfunc(
    types.int64(
        types.CPointer(types.CPointer(types.float64)),
        types.CPointer(types.CPointer(types.float64)),
        types.CPointer(types.CPointer(types.float64)),
        types.CPointer(types.float64),
        types.CPointer(types.CPointer(types.float64)),
        types.CPointer(types.CPointer(types.float64)),
        types.CPointer(types.float64),
        types.CPointer(types.int64),
        types.CPointer(types.int64),
        types.CPointer(types.int64),
        types.CPointer(types.int64),
        types.float64,
        types.int64,
    )
)
def heat_integrand(
    targets_, shapes_, mapping_, rst_, fields_, data_, tmp_, locations_, sizes_,
    shape_sizes_, field_sizes_, weight, element,
):  # fmt: skip
    sizes = carray(sizes_, 6)
    ndof, ndofpadded = sizes[3], sizes[4]
    tangent = carray(targets_[0], (ndof, ndofpadded))
    residual = carray(targets_[1], ndofpadded)
    shapes = carray(shapes_[0], (3, ndofpadded))
    dofs = data_[0]
    v, c1, latent, t_melt, t_sigma, k0, k1 = carray(data_[1], 7)

    T, dTdx, dTdy = 0.0, 0.0, 0.0
    for i in range(ndof):
        Ti = dofs[locations_[i]]
        T += shapes[0, i] * Ti
        dTdx += shapes[1, i] * Ti
        dTdy += shapes[2, i] * Ti

    # apparent heat capacity with the latent heat as the derivative of a tanh step
    th = np.tanh((T - t_melt) / t_sigma)
    sech2 = 1.0 - th * th
    c = 1.0 + c1 * T + latent * sech2 / (2.0 * t_sigma)
    dc = c1 - latent * sech2 * th / t_sigma**2
    k = k0 + k1 * T

    for i in range(ndof):
        flux = shapes[1, i] * dTdx + shapes[2, i] * dTdy
        residual[i] -= weight * (shapes[0, i] * c * v * dTdx + k * flux)
        for j in range(ndof):
            advection = v * (dc * shapes[0, j] * dTdx + c * shapes[1, j])
            diffusion = k * (shapes[1, i] * shapes[1, j] + shapes[2, i] * shapes[2, j])
            tangent[i, j] += weight * (
                shapes[0, i] * advection + k1 * shapes[0, j] * flux + diffusion
            )
    return 0


def to_scipy(matrix):
    data = np.asarray(matrix.data_array)
    indices = np.asarray(matrix.indices_array)
    indptr = np.asarray(matrix.indptr_array)
    return sp.csr_matrix((data, indices, indptr), shape=tuple(matrix.shape))


def assemble(integrand, constraints):
    matrix = mlhp.allocateSparseMatrix(basis, constraints[0])
    vector = mlhp.allocateRhsVector(matrix)
    targets = [matrix, vector]
    mlhp.integrateOnDomain(basis, integrand, targets, dirichletDofs=constraints)
    return to_scipy(matrix), np.asarray(vector).copy()


def assemble_flux(expression):
    vector = mlhp.allocateRhsVector(mlhp.allocateSparseMatrix(basis, dirichlet[0]))
    flux = mlhp.vectorField(2, [mlhp.scalarField(2, f"{flux_scale} * ({expression})")])
    mlhp.integrateOnSurface(
        basis, mlhp.neumannIntegrand(flux), [vector], top, dirichletDofs=dirichlet
    )
    return np.asarray(vector).copy()


def newton(T, load, K_pin=None):
    K_pin = sp.csr_matrix(M.shape) if K_pin is None else K_pin
    for _ in range(NEWTON_ITERS):
        dofs = mlhp.inflateDofs(mlhp.DoubleVector(T), dirichlet)
        integrand = mlhp.domainIntegrand(
            2,
            heat_integrand,
            types=[mlhp.AssemblyType.UnsymmetricMatrix, mlhp.AssemblyType.Vector],
            maxdiff=1,
            data=[dofs, parameters],
            abi=1,
        )
        tangent, residual = assemble(integrand, increment)
        lu = splu((tangent + K_pin).tocsc())
        dT = lu.solve(residual + load - K_pin @ T)
        T = T + dT
        if not np.isfinite(T).all():
            break
        if np.linalg.norm(dT) < NEWTON_TOL * np.linalg.norm(T):
            return T, lu, True
    return T, lu, False


def forward(load, depth=0):
    global T_warm, load_warm
    T, lu, converged = newton(T_warm, load)
    if not converged:
        # load continuation from the last converged state, bisecting towards the load
        assert depth < 20, "newton does not converge"
        forward(0.5 * (load_warm + load), depth + 1)
        return forward(load, depth + 1)
    T_warm, load_warm = T, load
    return T, lu


def loss_and_gradient(beta):
    T, lu = forward(B @ beta)
    residual = T - T_target
    lam = lu.solve(M @ residual, trans="T")
    return 0.5 * residual @ M @ residual / J0, B.T @ lam / J0


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
increment = [dirichlet[0], [0.0] * len(dirichlet[0])]

mass = mlhp.l2DomainIntegrand(
    rhs=mlhp.scalarField(2, 0.0), mass=mlhp.scalarField(2, 1.0)
)
M, _ = assemble(mass, dirichlet)

hat = "(1 - abs(x - {0}) / {1}) if abs(x - {0}) < {1} else 0"
B = np.stack([assemble_flux(hat.format(xi, spacing)) for xi in x_controls], axis=1)

T_cold = np.full(M.shape[0], T0)
beam_true = PEAK * np.exp(-((x_controls - CENTER) ** 2) / (2 * SIGMA**2))
if TARGET == "gaussian":
    gaussian = f"{PEAK} * exp(-(x - {CENTER})**2 / (2 * {SIGMA}**2))"
    T_target, _, converged = newton(T_cold, assemble_flux(gaussian))
else:
    x1, x2, depth = MELTPOOL
    inside = f"(x > {x1}) * (x < {x2}) * (y > {LENGTHS[1] - depth})"
    pin = mlhp.l2DomainIntegrand(
        rhs=mlhp.scalarField(2, f"{PENALTY * T_POOL} * {inside}"),
        mass=mlhp.scalarField(2, f"{PENALTY} * {inside}"),
    )
    K_pin, f_pin = assemble(pin, dirichlet)
    T_target, _, converged = newton(T_cold, f_pin, K_pin)
assert converged, "newton does not converge for the target"

residual0 = T_cold - T_target
J0 = 0.5 * residual0 @ M @ residual0  # unheated plate, reported losses are relative
# newton warm start, the last converged state across the optimizer iterations
T_warm, load_warm = T_cold, np.zeros_like(T_cold)

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
T_opt, _ = forward(B @ beta)
# each field keeps its own triangulation, the output cell order is thread dependent
fields = [temperature_field(T_target), temperature_field(T_opt)]
temperatures = np.concatenate([temperature for _, temperature in fields])
levels = np.linspace(temperatures.min(), temperatures.max(), 64)

fig, axes = plt.subplots(2, 1, figsize=(7, 3.2), dpi=200)
for ax, (tri, temperature) in zip(axes, fields):
    ax.tricontourf(tri, temperature, levels=levels, cmap="inferno")
    ax.tricontour(tri, temperature, levels=[T_LIQUIDUS], colors="w", linewidths=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.05)

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
    plt.savefig(RGB_PDF_DIR / f"beam_shape_nonlinear_{TARGET}.pdf")
    plt.close()
    beam_csv = CSV_DIR / f"beam_shape_nonlinear_{TARGET}_beam.csv"
    save_csv(beam_csv, x=x_controls, beta=beta, true=beam_true)
    save_csv(CSV_DIR / f"beam_shape_nonlinear_{TARGET}_loss.csv", loss=history)
