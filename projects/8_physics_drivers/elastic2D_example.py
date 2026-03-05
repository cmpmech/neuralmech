import mlhp

D = 2

print("1. Setting up mesh and basis", flush=True)

origin, max = [0.0] * D, [1.0] * D

cube = mlhp.implicitCube(origin, max)

# circle = mlhp.implicitSphere([0.2, 0.3], 0.1)
# circle = mlhp.implicitSphere([0.4, 0.3], 0.15)
circle = mlhp.implicitCube((0.2, 0.2), (0.4, 0.4))
cylinder1 = mlhp.extrude(circle, -1., 1., 2) if D == 3 else circle

cylinders = mlhp.implicitUnion([cylinder1])  # add the other cylinders
domain = mlhp.implicitSubtraction([cube, cylinders])

# Setup discretization
polynomialDegree = 2
nelements = [20] * D
alphaFCM = 1e-8

lengths = [m - o for o, m in zip(origin, max)]

baseGrid = mlhp.makeGrid(nelements, lengths, origin)
print(baseGrid)

grid = mlhp.makeRefinedGrid(mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=polynomialDegree + 2))
basis = mlhp.makeHpTrunkSpace(grid, degree=polynomialDegree, nfields=D)

print(grid)
print(basis)

print("2. Integrating Dirichlet boundary condition", flush=True)

# Face numbering: (2 * normalAxis + side): left -> 0, right -> 1, front -> 2, back -> 3, bottom -> 4, top -> 5
# Fix left edge in both directions
leftDofs = mlhp.integrateDirichletDofs(mlhp.vectorField(D, [0.0] * D), basis, [0])

dirichlet = mlhp.combineDirichletDofs([leftDofs])

print("3. Allocating linear system", flush=True)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

print("4. Computing domain integral", flush=True)

E = mlhp.scalarField(D, 210e9)
nu = mlhp.scalarField(D, 0.3)
rhs = mlhp.vectorField(D, [0.0] * D)

kinematics = mlhp.smallStrainKinematics(D)
constitutive = mlhp.planeStressMaterial(E, nu)
integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, rhs)

quadrature = mlhp.spaceTreeQuadrature(domain,
                                      depth=polynomialDegree + 1, epsilon=alphaFCM)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet)

print("5. Computing surface integral (traction on right edge)", flush=True)

traction = mlhp.vectorField(D, [1e6, 0.0])  # 1 MPa in x-direction
tractionIntegrand = mlhp.neumannIntegrand(traction)
tractionQuadrature = mlhp.quadratureOnMeshFaces(grid, [1])  # right face

mlhp.integrateOnSurface(basis, tractionIntegrand, [vector], tractionQuadrature, dirichletDofs=dirichlet)

print("6. Solving linear system", flush=True)

P = mlhp.diagonalPreconditioner(matrix)

internalDofs, norms = mlhp.cg(matrix, vector, rtol=1e-12, M=P, maxiter=2000, residualNorms=True)

allDofs = mlhp.inflateDofs(internalDofs, dirichlet)



# --------------------------- post-processing ----------------------------
import matplotlib.pyplot as plt
import cmasher as cmr

result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [5] * D)
processors = [mlhp.solutionProcessor(D, allDofs, "Displacement"),
              mlhp.vonMisesProcessor(allDofs, kinematics, constitutive, "VonMises")]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

# Displacement magnitude
tri = result.triangulation(mpl=True)
data = result.data()
import numpy as np
disp = np.array(data[0]).reshape(-1, D)

fig, ax = plt.subplots(figsize=(5,5), dpi=400)
ax.tricontourf(tri, disp[:,0], levels=64, cmap='turbo') # x displacement
# ax.tricontourf(tri, disp[:,0], levels=64, cmap=cmr.pride) # x displacement
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig('../../results/platewithahole.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
