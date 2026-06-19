import argparse
import os

# small system: single-threaded CHOLMOD/BLAS beats multithreaded spawn overhead
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.ndimage
from tqdm import tqdm

from solvers.optimization import MMA, ReferenceMMA, StructuredFEM

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_heat"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# volume-to-point heat conduction with uniform heat generation & heat sink

# geometry
LENGTHS = [1.0, 1.0]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 400  # 200  # 200  # 400  # 400
SUB_VOXELS = 1  # do not seem helpful
DEGREE = 1
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.4
PENAL = 3.0
RMIN = 2
K0, KMIN = (1.0, 1e-4)  # solid / void conductivity contrast
SOURCE = 1.0  # uniform volumetric heat generation
SINK_FRACTION = 0.1  # heat-sink length as a fraction of the left edge

# postprocessing
THRESHOLD = 0.5

# optimization with beta-continuation
MAX_ITER = 800
CHANGE_TOL = 0.01
MMA_MOVE = 0.2  # MMA step move limit
USE_DUAL = True  # True: dual MMA subsolver; False: mmapy (slower, but better)
ETA_E, ETA_I, ETA_D = 0.6, 0.5, 0.4  # eroded / intermediate / dilated thresholds
BETA_MAX = 16.0
CONT_STEP = 25

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
N_elems = nelx_e * nely_e
elem_lengths = [LENGTHS[0] / nelx_e, LENGTHS[1] / nely_e]

mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx_e, nely_e], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=1)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

# --------------------------- preintegrate reference element --------------------------
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=1)
ndof_e = basis_local.ndof()

integrand = mlhp.poissonIntegrand(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, 0.0))
quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS, SUB_VOXELS])
K_locals = mlhp.integratePartitionMatrices(
    basis_local, integrand, quadrature, mlhp.absoluteQuadratureOrder([QUAD_ORDER, 2])
)

# -------------------------------- boundary conditions --------------------------------
zero = mlhp.scalarField(2, 0.0)
left_dofs = set(mlhp.integrateDirichletDofs(zero, basis, [0])[0])
centroids = np.array(mesh.map(list(range(N_elems)), [[0.0, 0.0]] * N_elems))
left_col = centroids[:, 0] < elem_lengths[0]
central = np.abs(centroids[:, 1] - 0.5 * LENGTHS[1]) <= 0.5 * SINK_FRACTION * LENGTHS[1]
sink_cells = np.where(left_col & central)[0].tolist()
sink = sorted(left_dofs & set(mlhp.findSupportedDofs(basis, sink_cells)))

fixed = np.array(sink)
free = np.setdiff1d(np.arange(ndof), fixed)  # all non-sink dofs

# uniform volumetric source (assembled once)
no_bc = mlhp.combineDirichletDofs([])
m = mlhp.allocateSparseMatrix(basis, no_bc[0])
v = mlhp.allocateRhsVector(m)
source_domain = mlhp.implicitCube([0.0, 0.0], LENGTHS)
mlhp.integrateOnDomain(
    basis,
    mlhp.poissonIntegrand(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, SOURCE)),
    [m, v],
    quadrature=mlhp.spaceTreeQuadrature(source_domain, depth=QUAD_ORDER, epsilon=1e-8),
    dirichletDofs=no_bc,
)
force = np.asarray(v).copy()
force_free = force[free]

# --------------------------- FEM assembly & solver helpers ---------------------------
fem = StructuredFEM(efts, free, ndof, K_locals, (NX, NY), SUB_VOXELS)


def simp(rho):  # SIMP conductivity interpolation between void and solid
    return KMIN + rho**PENAL * (K0 - KMIN)


# --------------------------- density filter & Heaviside projection -------------------
ceil_r = int(np.ceil(RMIN))
ky, kx = np.meshgrid(np.arange(-ceil_r, ceil_r + 1), np.arange(-ceil_r, ceil_r + 1))
kernel = np.maximum(0.0, RMIN - np.sqrt(kx**2 + ky**2))
Hs = scipy.ndimage.convolve(np.ones((NX, NY)), kernel, mode="constant", cval=0.0)


def density_filter(x):  # conic smoothing of the raw design
    return scipy.ndimage.convolve(x, kernel, mode="constant", cval=0.0) / Hs


def filter_adjoint(g):  # transpose of density_filter, for the chain rule
    return scipy.ndimage.convolve(g / Hs, kernel, mode="constant", cval=0.0)


def projection(x_tilde, beta, eta):  # smoothed Heaviside about threshold eta
    a, b = np.tanh(beta * eta), np.tanh(beta * (1.0 - eta))
    return (a + np.tanh(beta * (x_tilde - eta))) / (a + b)


def dprojection(x_tilde, beta, eta):  # d(projection)/d(x_tilde)
    a, b = np.tanh(beta * eta), np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x_tilde - eta)) ** 2) / (a + b)


# ------------------------------------ optimization -----------------------------------
n = NX * NY
mma = MMA(n, move=MMA_MOVE) if USE_DUAL else ReferenceMMA(n, move=MMA_MOVE)
xval = np.full((n, 1), VOLFRAC)

beta = 1.0
c_ref = None
vfrac_d = VOLFRAC
history = []

ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    x_tilde = density_filter(xval.reshape(NX, NY))
    x_e = projection(x_tilde, beta, ETA_E)
    x_i = projection(x_tilde, beta, ETA_I)
    x_d = projection(x_tilde, beta, ETA_D)

    u = np.zeros(ndof)
    u[free] = fem.solve(simp(x_e), force_free)
    compliance = force @ u
    c_ref = compliance if c_ref is None else c_ref
    ce = fem.element_energy(u)
    dc = filter_adjoint(
        -PENAL
        * x_e ** (PENAL - 1)
        * (K0 - KMIN)
        * ce
        * dprojection(x_tilde, beta, ETA_E)
    )

    vfrac_d *= VOLFRAC / x_i.mean()  # intermediate -> VOLFRAC
    dvol = filter_adjoint(dprojection(x_tilde, beta, ETA_D) / n)

    f0val = compliance / c_ref
    df0dx = dc.ravel() / c_ref
    fval = x_d.mean() / vfrac_d - 1.0
    dfdx = dvol.ravel() / vfrac_d

    xnew = mma.step(xval, f0val, df0dx, fval, dfdx)
    change = float(np.abs(xnew - xval).max())
    xval = xnew
    history.append(compliance)
    pbar.set_postfix(
        {
            "c": f"{compliance:.3e}",
            "vol": f"{x_i.mean():.3f}",
            "beta": f"{beta:.0f}",
            "ch": f"{change:.2e}",
        }
    )
    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(x_i.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:d}.jpg")
        plt.close()

    if beta < BETA_MAX and it > 0 and it % CONT_STEP == 0:
        beta = min(2.0 * beta, BETA_MAX)
    elif beta >= BETA_MAX and change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
x_int = projection(density_filter(xval.reshape(NX, NY)), beta, ETA_I)
rho_thresh = (x_int > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(x_int), force_free)
compliance_phys = force @ u
u[free] = fem.solve(simp(rho_thresh), force_free)
compliance_thresh = force @ u
print(
    f"intermediate c {compliance_phys:.3e} vol {x_int.mean():.3f}\n"
    f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}"
)

# temperature field of the intermediate design
u[free] = fem.solve(simp(x_int), force_free)
postmesh = mlhp.domainCellMesh(source_domain, [DEGREE + 1] * 2)
temperature = mlhp.DataAccumulator()
mlhp.basisOutput(
    basis,
    cellmesh=postmesh,
    processors=[
        mlhp.solutionProcessor(2, mlhp.DoubleVector(u.tolist()), "Temperature")
    ],
    output=temperature,
)

fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(
    temperature.triangulation(mpl=True),
    temperature.data()[0],
    levels=64,
    cmap="inferno",
)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "topopt_heat_temp.png", bbox_inches="tight", pad_inches=0)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()

for field, name in ((x_int, "topopt_heat"), (rho_thresh, "topopt_heat_thresh")):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    if args.book:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / f"{name}.png")
        plt.close()
    elif not args.animate:
        plt.show()
    else:
        plt.close()
