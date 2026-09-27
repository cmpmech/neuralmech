import mlhp

D = 3

print("1. Setting up mesh and basis", flush=True)

# Setup triangulation domain
triangulation = mlhp.readStl("../csg.stl")
kdtree = mlhp.buildKdTree(triangulation)
domain = mlhp.rayIntersectionDomain(triangulation, tree=kdtree)

# Setup discretization
youngsModulus = 200 * 1e9
poissonsRatio = 0.3

polynomialDegree = 1
nelements = [50] * D
alphaFCM = 1e-5
penalty = 1e5 * youngsModulus

origin, max = triangulation.boundingBox()

# ##################### FIX ####################
origin = [o - 1e-10 for o in origin]
max = [m + 1e-10 for m in max]

lengths = [m - o for o, m in zip(origin, max)]

grid = mlhp.makeRefinedGrid(nelements, lengths, origin)
basis = mlhp.makeHpTensorSpace(grid, polynomialDegree, nfields=D)

print(basis)

print("2. Allocating linear system", flush=True)

matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)

print("2. Computing weak boundary integrals", flush=True)

def createBoundaryQuadrature(func):
    filtered = triangulation.filter(mlhp.implicitFunction(D, func))
    intersected, celldata = mlhp.intersectWithMesh(filtered, grid)
    quadrature = mlhp.simplexQuadrature(intersected, celldata)
    
    return intersected, celldata, quadrature
    
intersected0, celldata0, quadrature0 = createBoundaryQuadrature(f"x < {origin[0] + 1e-6}")
intersected1, celldata1, quadrature1 = createBoundaryQuadrature(f"x > {origin[0] + lengths[0] - 1e-6}")

integrand0 = mlhp.l2BoundaryIntegrand(mlhp.vectorField(D, [penalty] * D),
                                      mlhp.vectorField(D, [0.0] * D))

integrand1 = mlhp.neumannIntegrand(mlhp.vectorField(D, [1e3, 0.0, 0.0]))

mlhp.integrateOnSurface(basis, integrand0, [matrix, vector], quadrature0)
mlhp.integrateOnSurface(basis, integrand1, [vector], quadrature1)

print("4. Computing domain integral", flush=True)

E = mlhp.scalarField(D, 200 * 1e9)
nu = mlhp.scalarField(D, 0.3)
rhs = mlhp.vectorField(D, [0.0, 0.0, 0.0])

kinematics = mlhp.smallStrainKinematics(D) 
constitutive = mlhp.isotropicElasticMaterial(E, nu)
integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, rhs)

quadrature = mlhp.momentFittingQuadrature(domain, 
    depth=polynomialDegree, epsilon=alphaFCM)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature)

print("6. Solving linear system", flush=True)

#P = mlhp.additiveSchwarzPreconditioner(matrix, basis, dirichlet[0])
P = mlhp.diagonalPreconditioner(matrix)

dofs, norms = mlhp.cg(matrix, vector, M=P, maxiter=2000, residualNorms=True)

#print(f"cond K after domain integral: {numpy.linalg.cond(matrix.todense())}")
#import matplotlib.pyplot as plt
#plt.loglog(norms)
#plt.show()

print("7. Postprocessing solution", flush=True)

# Output solution on FCM mesh and boundary surface
gradient = mlhp.projectGradient(basis, dofs, quadrature)

processors = [mlhp.solutionProcessor(D, dofs, "Displacement"),
              mlhp.stressProcessor(gradient, kinematics, constitutive),
              mlhp.vonMisesProcessor(dofs, kinematics, constitutive, "VonMises1"),
              mlhp.vonMisesProcessor(gradient, kinematics, constitutive, "VonMises2"),
              mlhp.strainEnergyProcessor(gradient, kinematics, constitutive)]

intersected, celldata = mlhp.intersectWithMesh(triangulation, grid, tree=kdtree)

surfmesh = mlhp.localSimplexCellMesh(intersected, celldata)

output0 = mlhp.PVtuOutput(filename="outputs/linear_elasticity_fcm_stl_boundary")
output1 = mlhp.PVtuOutput(filename="outputs/linear_elasticity_fcm_stl_fcmmesh")

mlhp.basisOutput(basis, surfmesh, output0, processors)
mlhp.basisOutput(basis, output=output1, processors=processors)

# Output boundary surfaces
surfmesh0 = mlhp.localSimplexCellMesh(intersected0, celldata0)
surfmesh1 = mlhp.localSimplexCellMesh(intersected1, celldata1)

surfwriter0 = mlhp.VtuOutput(filename="outputs/linear_elasticity_fcm_stl_boundary0")
surfwriter1 = mlhp.VtuOutput(filename="outputs/linear_elasticity_fcm_stl_boundary1")

mlhp.meshOutput(grid, surfmesh0, surfwriter0, [])
mlhp.meshOutput(grid, surfmesh1, surfwriter1, [])
