import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = 2

# discretization
DEGREE = 3
NELEMENTS = [64] * D
ALPHAFCM = 1e-8

# physics
WAVENUMBER = 140.0  # k = omega / c
DAMPING = 10.0  # 10.0  # 1.0  # eta in (k^2 + i eta)

# -------------------------------------- geometry -------------------------------------
origin, maximum = [0.0] * D, [1.0] * D
cube = mlhp.implicitCube(origin, maximum)
hole = mlhp.implicitSphere([0.5, 0.5], 0.15)
domain = mlhp.implicitSubtraction([cube, hole])

# ------------------------------------ volume load ------------------------------------
# narrow Gaussian as point source
center, width, amplitude = [0.25, 0.25], 0.01, 1.0
radius2 = f"((x - {center[0]})**2 + (y - {center[1]})**2)"
source = mlhp.scalarField(D, f"{amplitude} * exp(-{radius2} / (2 * {width}**2))")

# ---------------------------------------- mesh ---------------------------------------
# fictitious-domain mesh: keep only cells that intersect the holed domain
baseGrid = mlhp.makeGrid(NELEMENTS, [1.0] * D, origin)
mesh = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=DEGREE + 2)
)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=2)
print(basis)

integrand = mlhp.helmholtzIntegrand(
    mlhp.scalarField(D, WAVENUMBER), mlhp.scalarField(D, DAMPING), source
)

# homogeneous Neumann boundary conditions
dirichlet = mlhp.combineDirichletDofs([])  # TODO is this really needed?

# -------------------------------------- assembly -------------------------------------
matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)

# integrate over the cut domain (finite-cell quadrature around the hole)
quadrature = mlhp.spaceTreeQuadrature(domain, depth=DEGREE + 1, epsilon=ALPHAFCM)
mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

# solve directly (indefinite, ill-conditioned by the cut cells -> no iterative solver);
# mkl pardiso is ~4x faster than superlu here and dominates the runtime over assembly
operator = sp.csr_matrix(
    (
        np.asarray(matrix.data_array),
        np.asarray(matrix.indices_array),
        np.asarray(matrix.indptr_array),
    ),
    shape=tuple(matrix.shape),
)
interiorDofs = pypardiso.spsolve(operator, np.asarray(vector))
allDofs = mlhp.inflateDofs(mlhp.DoubleVector(interiorDofs), dirichlet)

# ----------------------------------- postprocessing ----------------------------------
postmesh = mlhp.domainCellMesh(domain, [DEGREE + 2] * D)
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, [mlhp.solutionProcessor(D, allDofs)])

data = np.array(result.data()[0])
u_re, u_im = data[0::2], data[1::2]
phase = np.arctan2(u_im, u_re)  # phase shift in (-pi, pi]
tri = result.triangulation()

limit_re = np.max(np.abs(u_re))
limit_im = np.max(np.abs(u_im))
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.tricontourf(tri, u_re, cmap="seismic", levels=np.linspace(-limit_re, limit_re, 64))
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RESULTS_DIR / "helmholtz2D_amp.png")
    plt.close()
else:
    plt.show()

fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.tricontourf(tri, u_im, cmap="seismic", levels=np.linspace(-limit_im, limit_im, 64))
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RESULTS_DIR / "helmholtz2D_phase.png")
    plt.close()
else:
    plt.show()
