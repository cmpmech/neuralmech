import argparse
import os

# small system: single-threaded CHOLMOD/BLAS beats multithreaded spawn overhead
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cvxopt
import cvxopt.cholmod
import matplotlib.pyplot as plt
import mlhp
import nlopt
import numpy as np
import scipy.ndimage
import scipy.sparse
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_heatV3"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# volume-to-point heat conduction with uniform heat generation & heat sink

# geometry
LENGTHS = [1.0, 1.0]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 200  # 400  # 400
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

# post-processing
THRESHOLD = 0.5

# optimization with beta-continuation
MAX_ITER = 800
CHANGE_TOL = 0.01
ETA_E, ETA_I, ETA_D = 0.6, 0.5, 0.4  # eroded / intermediate / dilated thresholds
BETA_MAX = 16.0
CONT_STEP = 25

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
N_elems = nelx_e * nely_e
n_sub = SUB_VOXELS**2
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
# every element contributes ndof_e^2 entries to the same (iK, jK) locs of K each iter
iK = np.repeat(efts, ndof_e, axis=1).ravel()
jK = np.tile(efts, (1, ndof_e)).ravel()


def grid_to_elements(field):  # (NX, NY) -> (N_elems, n_sub)
    return (
        field.reshape(nelx_e, SUB_VOXELS, nely_e, SUB_VOXELS)
        .transpose(0, 2, 1, 3)
        .reshape(N_elems, n_sub)
    )


def elements_to_grid(field):  # (N_elems, n_sub) -> (NX, NY)
    return (
        field.reshape(nelx_e, nely_e, SUB_VOXELS, SUB_VOXELS)
        .transpose(0, 2, 1, 3)
        .reshape(NX, NY)
    )


# see topopt_mbb.py for explanation
def build_assemble_K_free():
    dof_map = np.full(ndof, -1)
    dof_map[free] = np.arange(free.size)
    keep = (dof_map[iK] >= 0) & (dof_map[jK] >= 0)  # entries with both dofs free
    ri, rj = dof_map[iK[keep]], dof_map[jK[keep]]
    order = np.lexsort((ri, rj))  # column-major order expected by CSC
    data_idx = np.flatnonzero(keep)[order]  # gather positions into K_e.ravel()
    ri, rj = ri[order], rj[order]
    first = np.empty(ri.size, dtype=bool)
    first[0] = True
    first[1:] = (ri[1:] != ri[:-1]) | (rj[1:] != rj[:-1])
    seg = np.flatnonzero(first)  # duplicate (row, col) group boundaries
    indices = ri[first].astype(np.int32)
    indptr = np.concatenate(
        [[0], np.cumsum(np.bincount(rj[first], minlength=free.size))]
    ).astype(np.int32)

    def assemble_K_free(k_grid):  # element conductivity field -> free-free matrix
        k_e = grid_to_elements(k_grid)
        K_e = np.einsum("es,sij->eij", k_e, K_locals, optimize=True)
        data = np.add.reduceat(K_e.ravel()[data_idx], seg)
        return scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=(free.size, free.size)
        )

    return assemble_K_free


assemble_K_free = build_assemble_K_free()


# for CHOLMOD: the SPD system's sparsity is factored symbolically once
K_free = assemble_K_free(np.full((NX, NY), K0))
A = cvxopt.spmatrix(
    cvxopt.matrix(K_free.data),
    cvxopt.matrix(K_free.indices.tolist()),
    cvxopt.matrix(np.repeat(np.arange(free.size), np.diff(K_free.indptr)).tolist()),
    (free.size, free.size),
)
factor = cvxopt.cholmod.symbolic(A)


def solve_free(k_grid):  # solve K(k) T = f for the free temperatures
    A.V = cvxopt.matrix(assemble_K_free(k_grid).data)
    cvxopt.cholmod.numeric(A, factor)
    b = cvxopt.matrix(force_free)
    cvxopt.cholmod.solve(factor, b)
    return np.array(b).ravel()


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
# V3 reference: nlopt's compiled MMA runs one persistent solve per beta-continuation
# stage (so its asymptote history is kept only within the stage); the dilated volume
# bound is rescaled between stages so the intermediate design hits VOLFRAC -- holding it
# fixed during a stage keeps the moving-target constraint from destabilizing MMA. Kept
# to compare nlopt's staged behavior against the continuous MMA in topopt_heat.py
n = int(NX * NY)
state = {"beta": 1.0, "c_ref": None, "vfrac_d": VOLFRAC, "it": 0}
history = []
ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None


def objective(x, grad):  # thermal compliance of the eroded design
    beta = state["beta"]
    x_tilde = density_filter(x.reshape(NX, NY))
    x_e = projection(x_tilde, beta, ETA_E)
    u = np.zeros(ndof)
    u[free] = solve_free(KMIN + x_e**PENAL * (K0 - KMIN))
    compliance = force @ u
    if state["c_ref"] is None:
        state["c_ref"] = compliance
    ue = u[efts]
    ce = elements_to_grid(np.einsum("ei,sij,ej->es", ue, K_locals, ue, optimize=True))
    dc = filter_adjoint(
        -PENAL
        * x_e ** (PENAL - 1)
        * (K0 - KMIN)
        * ce
        * dprojection(x_tilde, beta, ETA_E)
    )
    if grad.size > 0:
        grad[:] = (dc / state["c_ref"]).ravel()
    state["it"] += 1
    history.append(compliance)
    pbar.update(1)
    pbar.set_postfix({"c": f"{compliance:.3e}", "beta": f"{beta:.0f}"})
    if args.animate:
        x_i = projection(x_tilde, beta, ETA_I)
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(x_i.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{state['it']:d}.jpg")
        plt.close()
    return compliance / state["c_ref"]


def volume(x, grad):  # dilated volume <= rescaled bound
    beta = state["beta"]
    x_tilde = density_filter(x.reshape(NX, NY))
    x_d = projection(x_tilde, beta, ETA_D)
    if grad.size > 0:
        grad[:] = (
            filter_adjoint(dprojection(x_tilde, beta, ETA_D) / n) / state["vfrac_d"]
        ).ravel()
    return x_d.mean() / state["vfrac_d"] - 1.0


tic = time.time()
pbar = tqdm(total=MAX_ITER)
xval = np.full(n, VOLFRAC)
x_prev = xval.copy()
change = 1.0
while state["it"] < MAX_ITER and (state["beta"] < BETA_MAX or change > CHANGE_TOL):
    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(volume, 0.0)
    opt.set_lower_bounds(np.zeros(n))
    opt.set_upper_bounds(np.ones(n))
    opt.set_maxeval(min(CONT_STEP, MAX_ITER - state["it"]))
    xval = opt.optimize(xval)
    x_i = projection(density_filter(xval.reshape(NX, NY)), state["beta"], ETA_I)
    state["vfrac_d"] *= VOLFRAC / max(
        float(x_i.mean()), 1e-3
    )  # intermediate -> VOLFRAC
    change = float(np.max(np.abs(xval - x_prev)))
    x_prev = xval.copy()
    if state["beta"] < BETA_MAX:
        state["beta"] = min(2.0 * state["beta"], BETA_MAX)

beta = state["beta"]
it = state["it"]
toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ---------------------------------- post-processing ----------------------------------
x_int = projection(density_filter(xval.reshape(NX, NY)), beta, ETA_I)
rho_thresh = (x_int > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = solve_free(KMIN + x_int**PENAL * (K0 - KMIN))
compliance_phys = force @ u
u[free] = solve_free(KMIN + rho_thresh**PENAL * (K0 - KMIN))
compliance_thresh = force @ u
print(
    f"intermediate c {compliance_phys:.3e} vol {x_int.mean():.3f}\n"
    f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}"
)

# temperature field of the intermediate design
u[free] = solve_free(KMIN + x_int**PENAL * (K0 - KMIN))
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
        RESULTS_DIR / "topopt_heatV3_temp.png", bbox_inches="tight", pad_inches=0
    )
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()

for field, name in ((x_int, "topopt_heatV3"), (rho_thresh, "topopt_heatV3_thresh")):
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
