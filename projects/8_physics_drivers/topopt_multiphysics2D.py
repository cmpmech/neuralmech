import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import mlhp
import mmapy
import numpy as np
from tqdm import tqdm

from solvers.optimization import (
    DensityFilter,
    StructuredFEM,
    dprojection,
    dsimp,
    projection,
    simp,
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_multiphysics"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
parser.add_argument(
    "--bound", type=float, default=None, help="overrides STIFFNESS_BOUND (inf: none)"
)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# heat sink: minimum thermal compliance under an upper bound on the structural one.
# The plate is mounted on a wall along its left edge (clamped) that cools it only
# through a narrow contact in the middle (heat sink), heat is generated uniformly,
# and the free right end carries a downward point load.

# geometry
LENGTHS = [2.0, 1.0]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 200
DEGREE = 1
QUAD_ORDER = DEGREE + 1  # integration

# physics
VOLFRAC = 0.4
PENAL = 3.0
RMIN = 3
K0, KMIN = 1.0, 1e-3  # solid / void conductivity
E0, EMIN, NU = 1.0, 1e-9, 0.3  # solid / void stiffness
SOURCE = 1.0  # uniform volumetric heat generation
SINK_FRACTION = 0.1  # heat-sink length as a fraction of the left edge
LOAD = -1.0  # at the middle of the right edge

# structural compliance bound, as a fraction of the uniform initial design's
STIFFNESS_BOUND = 0.15 if args.bound is None else args.bound

# postprocessing
THRESHOLD = 0.5

# optimization with beta-continuation
MAX_ITER = 400
CHANGE_TOL = 0.01
MMA_MOVE = 0.2
ETA = 0.5  # projection threshold
BETA_MAX = 16.0
CONT_STEP = 40

# ---------------------------------------- mesh ---------------------------------------
N_elems = NX * NY
h = [LENGTHS[0] / NX, LENGTHS[1] / NY]
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[NX, NY], lengths=LENGTHS))
centroids = np.array(mesh.map(list(range(N_elems)), [[0.0, 0.0]] * N_elems))


# --------------------------- preintegrate reference element --------------------------
def element_matrices(integrand, nfields):
    local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=h))
    return mlhp.integratePartitionMatrices(
        mlhp.makeHpTensorSpace(local, degree=DEGREE, nfields=nfields),
        integrand,
        mlhp.gridQuadrature(nsubcells=[1, 1]),
        mlhp.absoluteQuadratureOrder([QUAD_ORDER, QUAD_ORDER]),
    )


# faces: 0=left, 1=right, 2=bottom, 3=top
def face_dofs(basis, face, ifield=0):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


# ------------------------------------ heat conduction --------------------------------
basis_t = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=1)
ndof_t = basis_t.ndof()
efts_t = np.array(basis_t.locationMaps())

left_col = centroids[:, 0] < h[0]
central = np.abs(centroids[:, 1] - 0.5 * LENGTHS[1]) <= 0.5 * SINK_FRACTION * LENGTHS[1]
sink_cells = np.where(left_col & central)[0].tolist()
sink = np.intersect1d(
    face_dofs(basis_t, 0), mlhp.findSupportedDofs(basis_t, sink_cells)
)
free_t = np.setdiff1d(np.arange(ndof_t), sink)

# uniform source, lumped onto the four nodes of each bilinear element
force_t = np.zeros(ndof_t)
np.add.at(force_t, efts_t, SOURCE * h[0] * h[1] / efts_t.shape[1])

integrand_t = mlhp.poissonIntegrand(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, 0.0))
fem_t = StructuredFEM(
    efts_t, free_t, ndof_t, element_matrices(integrand_t, 1), (NX, NY)
)

# -------------------------------------- elasticity -----------------------------------
basis_s = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=2)
ndof_s = basis_s.ndof()
efts_s = np.array(basis_s.locationMaps())

clamped = np.union1d(face_dofs(basis_s, 0, 0), face_dofs(basis_s, 0, 1))
free_s = np.setdiff1d(np.arange(ndof_s), clamped)

# the two right-column cells meeting at the middle share one node on the right edge
right_col = centroids[:, 0] > LENGTHS[0] - h[0]
middle = np.abs(centroids[:, 1] - 0.5 * LENGTHS[1]) < h[1]
load_dof = face_dofs(basis_s, 1, 1)
for cell in np.where(right_col & middle)[0].tolist():
    load_dof = np.intersect1d(load_dof, mlhp.findSupportedDofs(basis_s, [cell]))
assert load_dof.size == 1
force_s = np.zeros(ndof_s)
force_s[load_dof] = LOAD

material = mlhp.planeStressMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand_s = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
fem_s = StructuredFEM(
    efts_s, free_s, ndof_s, element_matrices(integrand_s, 2), (NX, NY)
)


def solve(fem, force, free, x, v_min, v_max):
    """state and compliance of the (projected) design x."""
    u = np.zeros(force.size)
    u[free] = fem.solve(simp(x, PENAL, v_min, v_max), force[free])
    return u, force @ u


# ------------------------------------ optimization -----------------------------------
density_filter = DensityFilter(RMIN, (NX, NY))
n = NX * NY
xval = np.full((n, 1), VOLFRAC)
xold1 = xold2 = xval.copy()
low, upp = np.zeros((n, 1)), np.ones((n, 1))
m = 2  # volume and structural compliance

_, ct_ref = solve(fem_t, force_t, free_t, xval.reshape(NX, NY), KMIN, K0)
_, cs_ref = solve(fem_s, force_s, free_s, xval.reshape(NX, NY), EMIN, E0)
cs_max = STIFFNESS_BOUND * cs_ref

beta = 1.0
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    x_tilde = density_filter(xval.reshape(NX, NY))
    x = projection(x_tilde, beta, ETA)
    chain = dprojection(x_tilde, beta, ETA)

    u_t, ct = solve(fem_t, force_t, free_t, x, KMIN, K0)
    u_s, cs = solve(fem_s, force_s, free_s, x, EMIN, E0)
    dct = -dsimp(x, PENAL, KMIN, K0) * fem_t.element_energy(u_t)
    dcs = -dsimp(x, PENAL, EMIN, E0) * fem_s.element_energy(u_s)

    # objective and constraints g <= 0, all normalized to order one
    f0val = ct / ct_ref
    df0dx = density_filter.adjoint(dct * chain).reshape(n, 1) / ct_ref
    fval = np.array([[x.mean() / VOLFRAC - 1.0], [cs / cs_max - 1.0]])
    dfdx = np.stack(
        [
            density_filter.adjoint(chain / n).ravel() / VOLFRAC,
            density_filter.adjoint(dcs * chain).ravel() / cs_max,
        ]
    )

    xnew, *_, low, upp = mmapy.mmasub(
        m, n, it + 1, xval, np.zeros((n, 1)), np.ones((n, 1)), xold1, xold2,
        f0val, df0dx, fval, dfdx, low, upp,
        1.0, np.zeros((m, 1)), 1e3 * np.ones((m, 1)), np.zeros((m, 1)), move=MMA_MOVE,
    )  # fmt: skip
    change = float(np.abs(xnew - xval).max())
    xold2, xold1, xval = xold1, xval, xnew
    pbar.set_postfix(
        {
            "ct": f"{f0val:.3e}",
            "cs": f"{cs / cs_max:.3f}",
            "vol": f"{x.mean():.3f}",
            "beta": f"{beta:.0f}",
            "ch": f"{change:.2e}",
        }
    )
    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(x.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
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
x = projection(density_filter(xval.reshape(NX, NY)), beta, ETA)
x_thresh = (x > THRESHOLD).astype(float)
for design, name in ((x, "projected  "), (x_thresh, "thresholded")):
    _, ct = solve(fem_t, force_t, free_t, design, KMIN, K0)
    _, cs = solve(fem_s, force_s, free_s, design, EMIN, E0)
    print(
        f"{name} ct {ct / ct_ref:.4e} cs {cs / cs_ref:.4e} "
        f"(bound {STIFFNESS_BOUND:.4e}) vol {design.mean():.3f}"
    )

# fields of the thresholded design, with the void left transparent
u_t, _ = solve(fem_t, force_t, free_t, x_thresh, KMIN, K0)
u_s, _ = solve(fem_s, force_s, free_s, x_thresh, EMIN, E0)
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(x_thresh.ravel("C").astype(np.float32)),
    nvoxels=[NX, NY],
    lengths=LENGTHS,
)
postmesh = mlhp.gridCellMesh([DEGREE + 2, DEGREE + 2])
fields = {}
for basis, u, name in ((basis_t, u_t, "temperature"), (basis_s, u_s, "uy")):
    acc = mlhp.DataAccumulator()
    processors = [
        mlhp.solutionProcessor(2, mlhp.DoubleVector(u.tolist()), "Solution"),
        mlhp.functionProcessor(indicator_field, "Indicator"),
    ]
    mlhp.basisOutput(basis, postmesh, acc, processors)
    values = np.array(acc.data()[0])
    indicator = np.array(acc.data()[1])
    tri = acc.triangulation()
    tri.set_mask(indicator[tri.triangles].mean(axis=1) < 0.5)
    fields[name] = (tri, values if name == "temperature" else values[1::2])

for name, cmap in (("temperature", cmr.torch), ("uy", "turbo")):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.tricontourf(*fields[name], cmap=cmap, levels=64)
    ax.set_xlim(0, LENGTHS[0])
    ax.set_ylim(0, LENGTHS[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if args.book:
        plt.savefig(RGB_PDF_DIR / f"topopt_multiphysics_{name}.pdf", transparent=True)
        plt.close()
    elif not args.animate:
        plt.show()
    else:
        plt.close()
