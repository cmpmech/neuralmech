import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DIM = 2

# discretization
DEGREE = 3
NELEMENTS = [64] * DIM
ALPHAFCM = 1e-8

# physics
WAVENUMBER = 140.0  # k = omega / c
DAMPING = 10.0
LENGTH = 1.0

# -------------------------------------- geometry -------------------------------------
origin, max = [0.0] * DIM, [LENGTH] * DIM

cube = mlhp.implicitCube(origin, max)
hole = mlhp.implicitSphere([0.5, 0.5], 0.15)
domain = mlhp.implicitSubtraction([cube, hole])

# ------------------------------------ volume load ------------------------------------
# narrow Gaussian as "point" source
center, width, amplitude = [0.25, 0.25], 0.01, 1.0
radius2 = f"((x - {center[0]})**2 + (y - {center[1]})**2)"
source = mlhp.scalarField(DIM, f"{amplitude} * exp(-{radius2} / (2 * {width}**2))")

# ---------------------------------------- mesh ---------------------------------------
lengths = [m - o for o, m in zip(origin, max)]

baseGrid = mlhp.makeGrid(NELEMENTS, lengths, origin)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=DEGREE + 2)
)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=2)
print(basis)

# -------------------------------- boundary conditions --------------------------------
# homogeneous Neumann boundary conditions (nothing to impose)

# -------------------------------------- assembly -------------------------------------
matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)

integrand = mlhp.helmholtzIntegrand(
    mlhp.scalarField(DIM, WAVENUMBER), mlhp.scalarField(DIM, DAMPING), source
)

quadrature = mlhp.spaceTreeQuadrature(domain, depth=DEGREE + 1, epsilon=ALPHAFCM)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature)

# --------------------------------------- solve ---------------------------------------
operator = sp.csr_matrix(
    (
        np.asarray(matrix.data_array),
        np.asarray(matrix.indices_array),
        np.asarray(matrix.indptr_array),
    ),
    shape=tuple(matrix.shape),
)
internalDofs = pypardiso.spsolve(operator, np.asarray(vector))
allDofs = mlhp.DoubleVector(internalDofs)

# ----------------------------------- postprocessing ----------------------------------
result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [DEGREE + 2] * DIM)
processors = [mlhp.solutionProcessor(DIM, allDofs)]
mlhp.basisOutput(basis, cellmesh, result, processors)

data = np.array(result.data()[0])
u_re, u_im = data[0::2], data[1::2]
amp = np.sqrt((u_re**2 + u_im**2))
phase = np.arctan2(u_im, u_re)  # phase shift in ]-pi, pi]
tri = result.triangulation()
# mask FCM cut-cell triangles inside the hole
cx, cy = tri.x[tri.triangles].mean(1), tri.y[tri.triangles].mean(1)
tri.set_mask((cx - 0.5)**2 + (cy - 0.5)**2 < 0.15**2)

limit_amp = np.max(amp)
amp_cmap = cmr.get_sub_cmap(cmr.fusion_r, 0.5, 1.0)  # red lobe of the wave colormap

fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.tricontourf(tri, amp, cmap=amp_cmap, levels=np.linspace(0, limit_amp, 64))
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "helmholtz2D_amp.pdf", transparent=True)
    plt.close()
else:
    plt.show()

fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.tricontourf(tri, phase, cmap=cmr.infinity, levels=np.linspace(-np.pi, np.pi, 64))
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "helmholtz2D_phase.pdf", transparent=True)
    plt.close()
else:
    plt.show()
