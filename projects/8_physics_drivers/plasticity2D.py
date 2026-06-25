import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

from postprocessing import load_cmap
from solvers.material_subroutines.j2 import ABI, NHISTORY, build

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DIM = 2

# discretization
DEGREE = 2
NELEMENTS = [60] * DIM  # 40
ALPHAFCM = 1e-8

# physics (steel, plane strain)
E = 210e9
NU = 0.3
YIELD = 250e6
HARDENING = E / 50.0  # linear isotropic hardening modulus
LENGTH = 1.0

# loading: traction ramped on the right face until the hole rim yields
TRACTION = 140e6
NSTEPS = 10
NEWTON_ITER = 20
NEWTON_TOL = 1e-8

# [E, nu, sigmaY, H, beta]; beta = 0 -> isotropic hardening
params = [E, NU, YIELD, HARDENING, 0.0]

# -------------------------------------- geometry -------------------------------------
origin, maximum = [0.0] * DIM, [LENGTH] * DIM

cube = mlhp.implicitCube(origin, maximum)
hole = mlhp.implicitCube((0.3, 0.3), (0.7, 0.7))
# hole = mlhp.implicitSphere([0.5 * LENGTH] * DIM, 0.15 * LENGTH)
domain = mlhp.implicitSubtraction([cube, hole])

# ------------------------------------- subroutine ------------------------------------
lib = build()

# ---------------------------------------- mesh ---------------------------------------
lengths = [m - o for o, m in zip(origin, maximum)]

baseGrid = mlhp.makeGrid(NELEMENTS, lengths, origin)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=DEGREE + 2)
)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=DIM)
print(basis)

# -------------------------------- boundary conditions --------------------------------
leftDofs = mlhp.integrateDirichletDofs(mlhp.vectorField(DIM, [0.0] * DIM), basis, [0])
dirichlet = mlhp.combineDirichletDofs([leftDofs])

quadrature = mlhp.spaceTreeQuadrature(domain, depth=DEGREE + 1, epsilon=ALPHAFCM)

kinematics = mlhp.smallStrainKinematics(DIM)

# fictitious linear-elastic material outside the domain keeps each tangent definite
elastic = mlhp.planeStrainMaterial(mlhp.scalarField(DIM, E), mlhp.scalarField(DIM, NU))

# ----------------------------------- load stepping -----------------------------------
history = mlhp.meshFunction(grid, [0.0] * NHISTORY)
dofs0 = mlhp.DoubleVector(basis.ndof(), 0.0)

for istep in range(NSTEPS):
    P = TRACTION * (istep + 1) / NSTEPS

    traction = mlhp.vectorField(DIM, [P, 0.0])
    tractionIntegrand = mlhp.neumannIntegrand(traction)
    tractionQuadrature = mlhp.quadratureOnMeshFaces(grid, [1])  # right face

    material = mlhp.constitutiveEquation(
        DIM,
        lib.material_address,
        fields=[history],
        symmetric=True,
        incremental=True,
        data=[params],
        abi=ABI,
    )

    dofs1 = dofs0.copy()
    norm0 = 1.0

    for inewton in range(NEWTON_ITER):
        matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
        vector = mlhp.allocateRhsVector(matrix)

        mlhp.integrateOnSurface(
            basis,
            tractionIntegrand,
            [vector],
            tractionQuadrature,
            dirichletDofs=dirichlet,
        )

        increment = mlhp.add(dofs1, dofs0, -1.0)
        plastic = mlhp.staticDomainIntegrand(kinematics, material, dofs=increment)
        fictitious = mlhp.staticDomainIntegrand(kinematics, elastic, dofs=dofs1)
        integrand = mlhp.selectIntegrand(domain, plastic, default=fictitious)

        mlhp.integrateOnDomain(
            basis,
            integrand,
            [matrix, vector],
            quadrature=quadrature,
            dirichletDofs=dirichlet,
        )

        norm1 = mlhp.norm(vector)
        norm0 = norm1 if inewton == 0 else norm0

        if norm1 <= max(NEWTON_TOL * norm0, 1e-11 * E):
            break

        P0 = mlhp.diagonalPreconditioner(matrix)
        solution = mlhp.cg(matrix, vector, rtol=1e-12, M=P0, maxiter=2000)
        dofs1 = mlhp.add(dofs1, mlhp.inflateDofs(solution, dirichlet))

    # commit the converged plastic state into the history
    updated = mlhp.meshFunctionStrainUpdate(
        history,
        basis,
        basis,
        dofs0,
        dofs1,
        lib.update_address,
        kinematics=kinematics,
        data=[params],
        abi=ABI,
    )
    history = mlhp.localL2Projection(grid, updated, DEGREE, quadrature=quadrature)
    dofs0 = dofs1

    print(f"step {istep + 1:2d} / {NSTEPS}: P = {P / 1e6:6.1f} MPa, newton {inewton}")

# ----------------------------------- postprocessing ----------------------------------
# smooth the element-local (discontinuous) history with a global, continuous L2
# projection before plotting, as in mlhp's j2_plasticity_compiled example
historyBasis = mlhp.makeHpTrunkSpace(grid, degree=1, nfields=1)
history = mlhp.globalL2Projection(historyBasis, history, quadrature=quadrature)

material = mlhp.constitutiveEquation(
    DIM,
    lib.material_address,
    fields=[history],
    symmetric=True,
    incremental=True,
    data=[params],
    abi=ABI,
)
zeroDofs = mlhp.DoubleVector(basis.ndof(), 0.0)

result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [DEGREE + 2] * DIM)
processors = [
    mlhp.solutionProcessor(DIM, dofs1, "Displacement"),
    mlhp.vonMisesProcessor(zeroDofs, kinematics, material, "VonMises"),
    mlhp.functionProcessor(history, "History"),
]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

tri = result.triangulation(mpl=True)
data = result.data()
disp = np.array(data[0]).reshape(-1, DIM)  # vector field: D components per node
vonMises = np.array(data[1])  # scalar: one value per node
plastic_strain = np.array(data[2]).reshape(-1, NHISTORY)[:, 12]
plastic_strain = np.maximum(
    plastic_strain, 0.0
)  # physically non-negative; clip projection undershoot

fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
ax.tricontourf(tri, plastic_strain, levels=64, cmap=rainbow)
# ax.tricontourf(tri, vonMises, levels=64, cmap=rainbow)
# ax.tricontourf(tri, disp[:, 0], levels=64, cmap=rainbow)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RESULTS_DIR / "plasticity2D.png")
    plt.close()
else:
    plt.show()
