import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import mlhp
import numpy as np

from postprocessing import load_cmap

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = 2
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

# discretization
DEGREE = 1
NELEMENTS = [20] * D
ALPHAFCM = 1e-8

# physics
KAPPA = 1.0
LENGTH = 1.0

# -------------------------------------- geometry -------------------------------------
origin, max = [0.0] * D, [LENGTH] * D

cube = mlhp.implicitCube(origin, max)
hole = mlhp.implicitSphere([0.4, 0.3], 0.15)
domain = mlhp.implicitSubtraction([cube, hole])

# ---------------------------------------- mesh ---------------------------------------
lengths = [m - o for o, m in zip(origin, max)]

baseGrid = mlhp.makeGrid(NELEMENTS, lengths, origin)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=DEGREE + 2)
)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=1)
print(basis)

# -------------------------------- boundary conditions --------------------------------
leftDofs = mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0])
rightDofs = mlhp.integrateDirichletDofs(mlhp.scalarField(D, 1.0), basis, [1])
dirichlet = mlhp.combineDirichletDofs([leftDofs, rightDofs])

# -------------------------------------- assembly -------------------------------------
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

rhs = mlhp.scalarField(D, 0.0)
integrand = mlhp.poissonIntegrand(mlhp.scalarField(D, KAPPA), rhs)

quadrature = mlhp.spaceTreeQuadrature(domain, depth=DEGREE + 1, epsilon=ALPHAFCM)

mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

# --------------------------------------- solve ---------------------------------------
P = mlhp.diagonalPreconditioner(matrix)
internalDofs, norms = mlhp.cg(
    matrix, vector, rtol=1e-12, M=P, maxiter=2000, residualNorms=True
)

allDofs = mlhp.inflateDofs(internalDofs, dirichlet)

# ----------------------------------- postprocessing ----------------------------------
result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [5] * D)
processors = [mlhp.solutionProcessor(D, allDofs, "Temperature")]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
cb = plt.tricontourf(
    result.triangulation(mpl=True), result.data()[0], levels=64, cmap=cmr.torch
)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig("../../results/poissonplate.pdf")
    plt.close()
else:
    plt.show()
