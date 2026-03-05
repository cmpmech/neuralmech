import mlhp

D = 2

print("1. Setting up mesh and basis", flush=True)

origin, max = [0.0] * D, [1.0] * D

cube = mlhp.implicitCube(origin, max)

# circle = mlhp.implicitSphere([0.2,0.3], 0.1)
circle = mlhp.implicitSphere([0.4, 0.3], 0.15)
cylinder1 = mlhp.extrude(circle, -1., 1., 2) if D == 3 else circle

cylinders = mlhp.implicitUnion([cylinder1]) # add the other cylinders
domain = mlhp.implicitSubtraction([cube, cylinders])

# Setup discretization
conductivity = 1.0
polynomialDegree = 1
nelements = [20] * D
alphaFCM = 1e-8

lengths = [m - o for o, m in zip(origin, max)]

baseGrid = mlhp.makeGrid(nelements, lengths, origin)
print(baseGrid)

grid = mlhp.makeRefinedGrid(mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=polynomialDegree + 2))
basis = mlhp.makeHpTrunkSpace(grid, degree=polynomialDegree, nfields=1)

print(grid)
print(basis)

print("2. Integrating Dirichlet boundary condition", flush=True)

# Face numbering: (2 * normalAxis + side): left -> 0, right -> 1, front -> 2, back -> 3, bottom -> 4, top -> 5
leftDofs = mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0])
rightDofs = mlhp.integrateDirichletDofs(mlhp.scalarField(D, 1.0), basis, [1])

dirichlet = mlhp.combineDirichletDofs([leftDofs, rightDofs])#

print("3. Allocating linear system", flush=True)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

print("4. Computing domain integral", flush=True)

rhs = mlhp.scalarField(D, 0.0)
integrand = mlhp.poissonIntegrand(mlhp.scalarField(D, conductivity), rhs)

quadrature = mlhp.spaceTreeQuadrature(domain,
                                      depth=polynomialDegree + 1, epsilon=alphaFCM)

# quadrature = mlhp.momentFittingQuadrature(domain,
#    depth=polynomialDegree + 3, epsilon=alphaFCM)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet)

print("6. Solving linear system", flush=True)

# P = mlhp.additiveSchwarzPreconditioner(matrix, basis, dirichlet[0])
P = mlhp.diagonalPreconditioner(matrix)

internalDofs, norms = mlhp.cg(matrix, vector, rtol=1e-12, M=P, maxiter=2000, residualNorms=True)

allDofs = mlhp.inflateDofs(internalDofs, dirichlet)

plotResult = True

# --------------------------- post-processing ----------------------------
import matplotlib.pyplot as plt
import cmasher as cmr
result = mlhp.DataAccumulator()
cellmesh = mlhp.domainCellMesh(domain, [5] * D)
processors = [mlhp.solutionProcessor(D, allDofs, "Solution")]
mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)

fig, ax = plt.subplots(figsize=(5,5), dpi=400)
cb = plt.tricontourf(result.triangulation(mpl=True), result.data()[0], levels=64, cmap=cmr.torch)
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig('../../results/poissonplate.pdf', bbox_inches='tight', pad_inches=0)
plt.show()