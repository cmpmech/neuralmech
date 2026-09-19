import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
from tqdm import tqdm

from solvers.optimization import DensityFilter, StructuredFEM, dsimp, simp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
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
NX, NY = np.array(LENGTHS).astype(int) * 150
SUB_VOXELS = 6
DEGREE = 3
QUAD_ORDER = DEGREE + 1  # integration

# physics
VOLFRAC = 0.5
PENAL = 3.0
RMIN = 2
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# postprocessing
THRESHOLD = 0.5

# optimization (optimality criterion)
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 500
CHANGE_TOL = 0.01

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
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
    basis_local,
    integrand,
    quadrature,
    mlhp.absoluteQuadratureOrder([QUAD_ORDER, QUAD_ORDER]),
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
free = np.setdiff1d(np.arange(ndof), fixed)  # all non-fixed dofs

force = np.zeros(ndof)
force[load_dof] = LOAD
force_free = force[free]

# --------------------------- FEM assembly & solver helpers ---------------------------
fem = StructuredFEM(efts, free, ndof, K_locals, (NX, NY), SUB_VOXELS)


# ----------------------------------- density filter ----------------------------------
density_filter = DensityFilter(RMIN, (NX, NY))

# ------------------------------------ optimization -----------------------------------
rho = np.full((NX, NY), VOLFRAC)
history = []

if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho, PENAL, EMIN, E0), force_free)
    compliance = force @ u

    # compliance sensitivity, mapped back to the design grid
    dc = -dsimp(rho, PENAL, EMIN, E0) * fem.element_energy(u)
    dc = density_filter.sensitivity(rho, dc)

    # optimality criterion update with bisection on the volume multiplier
    base = rho * np.maximum(0.0, -dc) ** DAMPING
    lo = np.maximum(0.0, rho - MOVE)
    hi = np.minimum(1.0, rho + MOVE)
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.clip(base / lmid**DAMPING, lo, hi)
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
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:d}.jpg")
        plt.close()

    if change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(rho_thresh, PENAL, EMIN, E0), force_free)
compliance_thresh = force @ u
print(f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}")

if not args.book and not args.animate:
    for field, name in ((rho, "topopt_mbb"), (rho_thresh, "topopt_mbb_thresh")):
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(
            field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

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
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "topopt_mbb_uy.pdf", transparent=True)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()
