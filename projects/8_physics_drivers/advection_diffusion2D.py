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

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DIM = 2

# discretization
DEGREE = 2
NELEMENTS = [40] * DIM
ALPHAFCM = 1e-8

# physics
KAPPA = 0.03  # diffusivity
VELOCITY = [0.25, 0.25]  # constant advection field
VELOCITY = [1, 1]  # constant advection field
LENGTH = 1.5

# gaussian source
AMPLITUDE = 1.0
CENTER = [0.15, 0.15]
WIDTH = 0.025

# -------------------------------------- geometry -------------------------------------
origin, maximum = [0.0] * DIM, [LENGTH] * DIM

domain = mlhp.implicitCube(origin, maximum)

# ------------------------------------- volume load -----------------------------------
velocity = mlhp.vectorField(DIM, VELOCITY)
diffusivity = mlhp.scalarField(DIM, KAPPA)

radius2 = f"((x - {CENTER[0]})**2 + (y - {CENTER[1]})**2)"
source = mlhp.scalarField(DIM, f"{AMPLITUDE} * exp(-{radius2} / (2 * {WIDTH}**2))")

# ---------------------------------------- mesh ---------------------------------------
lengths = [m - o for o, m in zip(origin, maximum)]

baseGrid = mlhp.makeGrid(NELEMENTS, lengths, origin)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=DEGREE + 2)
)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=1)
print(basis)

# -------------------------------- boundary conditions --------------------------------
# dirichlet at inflow, neumann at outflow
dirichlet = mlhp.integrateDirichletDofs(mlhp.scalarField(DIM, 0.0), basis, [0, 2])

# -------------------------------------- assembly -------------------------------------
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

integrand = mlhp.advectionDiffusionIntegrand(velocity, diffusivity, source)

quadrature = mlhp.spaceTreeQuadrature(domain, depth=DEGREE + 1, epsilon=ALPHAFCM)

mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

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

allDofs = mlhp.inflateDofs(mlhp.DoubleVector(internalDofs), dirichlet)

# ----------------------------------- postprocessing ----------------------------------
result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [DEGREE + 2] * DIM)
processors = [mlhp.solutionProcessor(DIM, allDofs, "Concentration")]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
ax.tricontourf(
    result.triangulation(mpl=True), result.data()[0], levels=64, cmap=cmr.torch
)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RESULTS_DIR / "advectiondiffusion.png")
    plt.close()
else:
    plt.show()
