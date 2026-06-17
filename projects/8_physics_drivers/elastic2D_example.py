import matplotlib.pyplot as plt
import mlhp
import numpy as np

# -------------------------------------- settings -------------------------------------
D = 2
polynomialDegree = 2
nelements = [20] * D
alphaFCM = 1e-8


# -------------------------------------- geometry -------------------------------------
origin, max = [0.0] * D, [1.0] * D

cube = mlhp.implicitCube(origin, max)
circle = mlhp.implicitCube((0.2, 0.2), (0.4, 0.4))
cylinder1 = mlhp.extrude(circle, -1.0, 1.0, 2) if D == 3 else circle

cylinders = mlhp.implicitUnion([cylinder1])  # add the other cylinders
domain = mlhp.implicitSubtraction([cube, cylinders])

# ----------------------------------- discretzation -----------------------------------
lengths = [m - o for o, m in zip(origin, max)]

baseGrid = mlhp.makeGrid(nelements, lengths, origin)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=polynomialDegree + 2)
)
basis = mlhp.makeHpTrunkSpace(grid, degree=polynomialDegree, nfields=D)
print(basis)

# -------------------------------- boundary conditions --------------------------------
# Face numbering: (2 * normalAxis + side): left -> 0, right -> 1, front -> 2, back -> 3, bottom -> 4, top -> 5
leftDofs = mlhp.integrateDirichletDofs(mlhp.vectorField(D, [0.0] * D), basis, [0])
dirichlet = mlhp.combineDirichletDofs([leftDofs])

# -------------------------------------- assembly -------------------------------------
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

E = mlhp.scalarField(D, 210e9)
nu = mlhp.scalarField(D, 0.3)
rhs = mlhp.vectorField(D, [0.0] * D)

kinematics = mlhp.smallStrainKinematics(D)
constitutive = mlhp.planeStressMaterial(E, nu)
integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, rhs)

quadrature = mlhp.spaceTreeQuadrature(
    domain, depth=polynomialDegree + 1, epsilon=alphaFCM
)

mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

# force
traction = mlhp.vectorField(D, [1e6, 0.0])
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

# ---------------------------------- post-processing ----------------------------------
result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [5] * D)
processors = [
    mlhp.solutionProcessor(D, allDofs, "Displacement"),
    mlhp.vonMisesProcessor(allDofs, kinematics, constitutive, "VonMises"),
]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

tri = result.triangulation(mpl=True)
data = result.data()
disp = np.array(data[0]).reshape(-1, D)

fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
ax.tricontourf(tri, disp[:, 0], levels=64, cmap="turbo")  # x displacement
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig("../../results/platewithahole.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
