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
import numpy as np
import scipy.ndimage
import scipy.sparse
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_mbb"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# half MBB beam (left edge is the symmetry plane)
# geometry
LENGTHS = [4.0, 1.0]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 150  # 90
SUB_VOXELS = 6
DEGREE = 3
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.5
PENAL = 3.0
RMIN = 2
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# post-processing
THRESHOLD = 0.5

# optimization
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 500
CHANGE_TOL = 0.01

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
N_elems = nelx_e * nely_e
n_sub = SUB_VOXELS**2
elem_lengths = [LENGTHS[0] / nelx_e, LENGTHS[1] / nely_e]

mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx_e, nely_e], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=2)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

# --------------------------- preintegrate reference element --------------------------
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=2)
ndof_e = basis_local.ndof()

material = mlhp.planeStressMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS, SUB_VOXELS])
K_locals = mlhp.integratePartitionMatrices(
    basis_local, integrand, quadrature, mlhp.absoluteQuadratureOrder([QUAD_ORDER, 2])
)


# -------------------------------- boundary conditions --------------------------------
# faces: 0=left, 1=right, 2=bottom, 3=top
def face_dofs(face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


symmetry = face_dofs(0, 0)  # left edge
roller = np.intersect1d(face_dofs(2, 1), face_dofs(1, 1))  # bottom-right corner
load_dof = np.intersect1d(face_dofs(3, 1), face_dofs(0, 1))  # top-left corner

fixed = np.unique(np.concatenate([symmetry, roller]))
free = np.setdiff1d(np.arange(ndof), fixed)  # all none fixed dofs

force = np.zeros(ndof)
force[load_dof] = LOAD
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


# def assemble_K_free(rho_field):  # penalised stiffness restricted to the free dofs
#     E_e = EMIN + grid_to_elements(rho_field) ** PENAL * (E0 - EMIN)
#     K_e = np.einsum("es,sij->eij", E_e, K_locals, optimize=True)
#     K = scipy.sparse.csc_matrix((K_e.ravel(), (iK, jK)), shape=(ndof, ndof))
#     K_free = K[free][:, free]
#     K_free.sort_indices()  # guarantees same ordering for repeated assemblies/solves
#     return K_free


# faster equivalent solution to assemble_K_free by Claude (but not understood)
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

    def assemble_K_free(rho_field):  # penalised stiffness restricted to the free dofs
        E_e = EMIN + grid_to_elements(rho_field) ** PENAL * (E0 - EMIN)
        K_e = np.einsum("es,sij->eij", E_e, K_locals, optimize=True)
        data = np.add.reduceat(K_e.ravel()[data_idx], seg)
        return scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=(free.size, free.size)
        )

    return assemble_K_free


assemble_K_free = build_assemble_K_free()


# for CHOLMOD: the SPD system's sparsity is factored symbolically once
K_free = assemble_K_free(np.full((NX, NY), VOLFRAC))
A = cvxopt.spmatrix(
    cvxopt.matrix(K_free.data),
    cvxopt.matrix(K_free.indices.tolist()),
    cvxopt.matrix(np.repeat(np.arange(free.size), np.diff(K_free.indptr)).tolist()),
    (free.size, free.size),
)
factor = cvxopt.cholmod.symbolic(A)


def solve_free(rho_field):
    A.V = cvxopt.matrix(assemble_K_free(rho_field).data)
    cvxopt.cholmod.numeric(A, factor)
    b = cvxopt.matrix(force_free)
    cvxopt.cholmod.solve(factor, b)
    return np.array(b).ravel()


# ----------------------------------- density filter ----------------------------------
ceil_r = int(np.ceil(RMIN))
ky, kx = np.meshgrid(np.arange(-ceil_r, ceil_r + 1), np.arange(-ceil_r, ceil_r + 1))
kernel = np.maximum(0.0, RMIN - np.sqrt(kx**2 + ky**2))
Hs = scipy.ndimage.convolve(np.ones((NX, NY)), kernel, mode="constant", cval=0.0)


def filter_sensitivity(rho, dc):
    num = scipy.ndimage.convolve(rho * dc, kernel, mode="constant", cval=0.0)
    return num / (np.maximum(rho, 1e-3) * Hs)


# ------------------------------------ optimization -----------------------------------
rho = np.full((NX, NY), VOLFRAC)
history = []

ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for iter in pbar:  # range(1, MAX_ITER + 1):
    u = np.zeros(ndof)
    u[free] = solve_free(rho)
    compliance = force @ u

    # compliance sensitivity, mapped back to the design grid
    ue = u[efts]
    ce = np.einsum("ei,sij,ej->es", ue, K_locals, ue, optimize=True)
    dc = -PENAL * rho ** (PENAL - 1) * (E0 - EMIN) * elements_to_grid(ce)
    dc = filter_sensitivity(rho, dc)

    # optimality-criterion update with bisection on the volume multiplier
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.maximum(
            0.0,
            np.maximum(
                rho - MOVE,
                np.minimum(
                    1.0,
                    np.minimum(
                        rho + MOVE, rho * (np.maximum(0.0, -dc) / lmid) ** DAMPING
                    ),
                ),
            ),
        )
        if rho_new.mean() > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid

    change = np.abs(rho_new - rho).max()
    rho = rho_new
    history.append(compliance)
    pbar.set_postfix({"c": f"{compliance:.3e}", "change": f"{change:.2e}"})

    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{iter:d}.jpg")
        plt.close()

    if change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {iter} iter\n"
    f"time per iter {(toc - tic) / iter:.2e} s"
)

# ---------------------------------- post-processing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = solve_free(rho_thresh)
compliance_thresh = force @ u
print(f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}")

for field, name in ((rho, "topopt_mbb"), (rho_thresh, "topopt_mbb_thresh")):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    if args.book:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / f"{name}.png", transparent=True)
        plt.close()
    elif not args.animate:
        plt.show()
    else:
        plt.close()

# y-displacement evaluated on the thresholded structure (void left transparent)
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(rho_thresh.ravel("C").astype(np.float32)),
    nvoxels=[NX, NY],
    lengths=LENGTHS,
)
processors = [
    mlhp.solutionProcessor(2, mlhp.DoubleVector(u.tolist()), "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2, DEGREE + 2])
acc = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, acc, processors)

uy = np.array(acc.data()[0])[1::2]
indicator = np.array(acc.data()[1])
tri = acc.triangulation()
tri.set_mask(indicator[tri.triangles].mean(axis=1) < 0.5)

fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(tri, uy, cmap="turbo", levels=64)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0)
if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "topopt_mbb_uy.png", transparent=True)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()
