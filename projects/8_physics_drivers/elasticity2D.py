import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

from postprocessing import load_cmap

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DIM = 2

# discretization
DEGREE = 2
NELEMENTS = [20] * DIM
ALPHAFCM = 1e-8

# physics
E = 210e9
NU = 0.3
LENGTH = 1.0

# -------------------------------------- geometry -------------------------------------
origin, max = [0.0] * DIM, [LENGTH] * DIM

cube = mlhp.implicitCube(origin, max)
hole = mlhp.implicitCube((0.2, 0.2), (0.4, 0.4))
domain = mlhp.implicitSubtraction([cube, hole])

# ---------------------------------------- mesh ---------------------------------------
lengths = [m - o for o, m in zip(origin, max)]

baseGrid = mlhp.makeGrid(NELEMENTS, lengths, origin)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=DEGREE + 2)
)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=DIM)
print(basis)

# -------------------------------- boundary conditions --------------------------------
leftDofs = mlhp.integrateDirichletDofs(mlhp.vectorField(DIM, [0.0] * DIM), basis, [0])
dirichlet = mlhp.combineDirichletDofs([leftDofs])

# -------------------------------------- assembly -------------------------------------
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

E_field = mlhp.scalarField(DIM, E)
nu_field = mlhp.scalarField(DIM, NU)
rhs = mlhp.vectorField(DIM, [0.0] * DIM)

kinematics = mlhp.smallStrainKinematics(DIM)
constitutive = mlhp.planeStressMaterial(E_field, nu_field)
integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, rhs)

quadrature = mlhp.spaceTreeQuadrature(domain, depth=DEGREE + 1, epsilon=ALPHAFCM)

mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

# force
traction = mlhp.vectorField(DIM, [1e6, 0.0])
tractionIntegrand = mlhp.neumannIntegrand(traction)
tractionQuadrature = mlhp.quadratureOnMeshFaces(grid, [1])  # right face

mlhp.integrateOnSurface(
    basis, tractionIntegrand, [vector], tractionQuadrature, dirichletDofs=dirichlet
)

# --------------------------------------- solve ---------------------------------------
P = mlhp.diagonalPreconditioner(matrix)
internalDofs, norms = mlhp.cg(
    matrix, vector, rtol=1e-12, M=P, maxiter=2000, residualNorms=True
)

allDofs = mlhp.inflateDofs(internalDofs, dirichlet)

# ----------------------------------- postprocessing ----------------------------------
# data preparation mlhp
result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [5] * DIM)
processors = [
    mlhp.solutionProcessor(DIM, allDofs, "Displacement"),
    mlhp.vonMisesProcessor(allDofs, kinematics, constitutive, "VonMises"),
]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

# data preparation numpy
tri = result.triangulation(mpl=True)
# mask FCM cut-cell triangles inside the hole
cx, cy = tri.x[tri.triangles].mean(1), tri.y[tri.triangles].mean(1)
tri.set_mask((cx > 0.2) & (cx < 0.4) & (cy > 0.2) & (cy < 0.4))
data = result.data()
disp = np.array(data[0]).reshape(-1, DIM)  # vector field: D components per node
stress = np.array(data[1])  # von Mises is a scalar field: one value per node


fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
ax.tricontourf(tri, disp[:, 0], levels=64, cmap="turbo")  # x displacement
# ax.tricontourf(tri, stress, levels=64, cmap=rainbow)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "platewithahole.pdf", transparent=True)
    plt.close()
else:
    plt.show()
