import argparse
import os

# small system: single-threaded BLAS beats multithreaded spawn overhead
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from solvers.optimization import StructuredFEM

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_heat_adam"

torch.manual_seed(2)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# volume-to-point heat conduction with uniform heat generation & heat sink,
# unconstrained Adam variant: distribute conductor in the full domain to minimize
# thermal compliance. no volume constraint -- compliance is minimized directly with
# Adam and the design sharpened by beta-continuation

# geometry
LENGTHS = [1.0, 1.0]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 400  # 200  # 200  # 400  # 400
SUB_VOXELS = 1
DEGREE = 1
QUAD_ORDER = DEGREE + 1

# physics
PENAL = 4.0
RMIN = 2
K0, KMIN = 1.0, 1e-4
SOURCE = 1.0
SINK_FRACTION = 0.1

# postprocessing
THRESHOLD = 0.5

# optimization with Adam and beta-continuation
MAX_ITER = 500
LR = 1e-1  # 5e-2
INITIAL_GUESS = 0.5
ETA = 0.5
BETA_MAX = 32.0
CONT_STEP = 25
VOLFRAC = 0.4
PENALTY0, PENALTY_INC, PENALTY_MAX = 0.1, 0.05, 100.0  # volume penalty continuation

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
free = np.setdiff1d(np.arange(ndof), fixed)

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
# Adam on the raw design with the adjoint gradient set by hand (the real sparse solve
# is not autodifferentiable). thermal compliance (normalized by c_ref) is minimized
# subject to a soft quadratic volume penalty penalty*(rho.mean()/VOLFRAC - 1)^2.
# the penalty weight grows from PENALTY0 to PENALTY_MAX each iteration so compliance
# drives the early shape and the volume constraint tightens progressively. the design
# is held in [0, 1] by clamping; beta-continuation sharpens it towards 0/1
n = NX * NY
x = torch.full(
    (n,), INITIAL_GUESS, dtype=torch.float64, device=device, requires_grad=True
)
optimizer = torch.optim.Adam([x], lr=LR)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=MAX_ITER, eta_min=LR * 1e-2
# )
scheduler = None

beta = 1.0
penalty = PENALTY0
c_ref = None
history = []

ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    x_tilde = density_filter(x.detach().numpy().reshape(NX, NY))
    rho = projection(x_tilde, beta, ETA)

    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho), force_free)
    compliance = float(force @ u)
    c_ref = compliance if c_ref is None else c_ref
    ce = fem.element_energy(u)
    dc = filter_adjoint(
        -PENAL * rho ** (PENAL - 1) * (K0 - KMIN) * ce * dprojection(x_tilde, beta, ETA)
    )
    vol_err = rho.mean() / VOLFRAC - 1.0
    dvol = filter_adjoint(
        2 * penalty * vol_err / (VOLFRAC * n) * dprojection(x_tilde, beta, ETA)
    )
    dc_total = dc / c_ref + dvol

    optimizer.zero_grad()
    x.grad = torch.from_numpy(dc_total.ravel())
    optimizer.step()
    with torch.no_grad():
        x.clamp_(0.0, 1.0)
    if scheduler is not None:
        scheduler.step()

    penalty = min(penalty + PENALTY_INC, PENALTY_MAX)
    history.append(compliance)
    pbar.set_postfix(
        {
            "c": f"{compliance:.3e}",
            "vol": f"{rho.mean():.3f}",
            "beta": f"{beta:.0f}",
            "pen": f"{penalty:.1f}",
        }
    )

    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:04d}.jpg")
        plt.close()

    if beta < BETA_MAX and it > 0 and it % CONT_STEP == 0:
        beta = min(2.0 * beta, BETA_MAX)

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
x_int = projection(density_filter(x.detach().numpy().reshape(NX, NY)), beta, ETA)
rho_thresh = (x_int > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(x_int), force_free)
compliance_phys = float(force @ u)

u_thresh = np.zeros(ndof)
u_thresh[free] = fem.solve(simp(rho_thresh), force_free)
compliance_thresh = float(force @ u_thresh)
print(
    f"intermediate c {compliance_phys:.3e} vol {x_int.mean():.3f}\n"
    f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}"
)

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
    plt.savefig(
        RESULTS_DIR / "topopt_heat_adam_temp.png", bbox_inches="tight", pad_inches=0
    )
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()

for field, name in (
    (x_int, "topopt_heat_adam"),
    (rho_thresh, "topopt_heat_adam_thresh"),
):
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
