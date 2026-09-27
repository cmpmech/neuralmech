import time

import mlhp
import numpy as np

D = 3

print("1. Setting up mesh and basis", flush=True)

origin, max = [-0.78] * D, [0.78] * D

sphere = mlhp.implicitSphere([0.0, 0.0, 0.0], 1.0)
cube = mlhp.implicitCube(origin, max)
intersection = mlhp.implicitIntersection([sphere, cube])

circle = mlhp.implicitSphere([0, 0], 0.4)

cylinder1 = mlhp.extrude(circle, -1.0, 1.0, 0)
cylinder2 = mlhp.extrude(circle, -1.0, 1.0, 1)
cylinder3 = mlhp.extrude(circle, -1.0, 1.0, 2)

cylinders = mlhp.implicitUnion([cylinder1, cylinder2, cylinder3])
domain = mlhp.implicitSubtraction([intersection, cylinders])

scaling = 2 * 2.5 / 0.78

domain = mlhp.implicitTransformation(domain, mlhp.scaling([scaling] * D))

# Setup discretization
youngsModulus = 200 * 1e9
poissonsRatio = 0.3

DEGREE = 1
REFINEMENT = 0
# N = 140
N = 160

nelements = [N] * D  # 100
alphaFCM = 1e-4  # doesn't really influence it -> 1e-8
penalty = 1e5 * youngsModulus

# ##################### FIX ####################
origin = [scaling * o - 1e-10 for o in origin]
max = [scaling * m + 1e-10 for m in max]

lengths = [m - o for o, m in zip(origin, max)]

grid = mlhp.makeGrid(nelements, lengths, origin)
print(grid)

grid = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(grid, domain=domain, nseedpoints=DEGREE + 2)
)
grid.refine(mlhp.refineTowardsBoundary(domain, REFINEMENT))

basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=D)

print(grid)
print(basis)

resolution = [DEGREE + 3] * D
triangulation, celldata = mlhp.recoverDomainBoundary(grid, domain, resolution)

print(basis)

print("2. Allocating linear system", flush=True)

matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)

print("3. Computing weak boundary integrals", flush=True)


def createBoundaryQuadrature(func):
    filteredTriangulation, filteredCelldata = triangulation.filter(
        mlhp.implicitFunction(D, func), celldata
    )
    quadrature = mlhp.simplexQuadrature(filteredTriangulation, filteredCelldata)

    return filteredTriangulation, filteredCelldata, quadrature


intersected0, celldata0, quadrature0 = createBoundaryQuadrature(
    f"x < {origin[0] + 1e-3}"
)
intersected1, celldata1, quadrature1 = createBoundaryQuadrature(
    f"x > {origin[0] + lengths[0] - 1e-3}"
)

integrand0 = mlhp.l2BoundaryIntegrand(
    mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, [0.0] * D)
)

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

# quadrature = mlhp.spaceTreeQuadrature(
#     domain, depth=polynomialDegree + 1, epsilon=alphaFCM
# )

# quadrature = mlhp.spaceTreeQuadrature(
#     domain, depth=polynomialDegree + 2, epsilon=alphaFCM
# )

# quadrature = mlhp.momentFittingQuadrature(
#     domain, depth=polynomialDegree + 3, epsilon=alphaFCM
# )

quadrature = mlhp.momentFittingQuadrature(domain, depth=DEGREE + 1, epsilon=alphaFCM)


mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature)

print("6. Solving linear system", flush=True)

P = mlhp.additiveSchwarzPreconditioner(matrix, basis)  # , dirichlet[0])
# P = mlhp.diagonalPreconditioner(matrix)

tic = time.time()
dofs, norms = mlhp.cg(matrix, vector, rtol=1e-12, M=P, maxiter=2000, residualNorms=True)

print(f"{len(norms)} CG iters")
print(f"elapsed solve time {time.time() - tic:.2f} s")

# Compliance = external work 1/2 f.u; physical scalar, robust FCM convergence metric
compliance = 0.5 * np.dot(np.array(vector.array), np.array(dofs.array))
print(f"compliance {compliance:.6e}")

# print(f"cond K after domain integral: {numpy.linalg.cond(matrix.todense())}")
# import matplotlib.pyplot as plt
# plt.loglog(norms)
# plt.show()

# print("7. Postprocessing solution", flush=True)

# # Output solution on FCM mesh and boundary surface
# gradient = mlhp.projectGradient(basis, dofs, quadrature)

# # Indicator: 1 inside the physical domain (material), 0 in the fictitious domain (void)
# indicator = domain.asfield(0.0, 1.0)

# # Max displacement restricted to material points (void DOFs are spurious under FCM)
# sampleAccumulator = mlhp.DataAccumulator()
# mlhp.basisOutput(
#     basis,
#     mlhp.gridCellMesh([polynomialDegree + 2] * D),
#     sampleAccumulator,
#     [mlhp.solutionProcessor(D, dofs, "Displacement"), mlhp.functionProcessor(indicator, "Indicator")],
# )
# sampleDisplacement = np.array(sampleAccumulator.data()[0]).reshape(-1, D)
# materialMask = np.array(sampleAccumulator.data()[1]) >= 0.5
# maxDisplacement = np.linalg.norm(sampleDisplacement[materialMask], axis=1).max()
# print(f"maximum displacement (material) {maxDisplacement:.6e}")

# processors = [
#     mlhp.solutionProcessor(D, dofs, "Displacement"),
#     mlhp.stressProcessor(gradient, kinematics, constitutive),
#     mlhp.vonMisesProcessor(dofs, kinematics, constitutive, "VonMises1"),
#     mlhp.vonMisesProcessor(gradient, kinematics, constitutive, "VonMises2"),
#     mlhp.strainEnergyProcessor(gradient, kinematics, constitutive),
#     mlhp.functionProcessor(indicator, "Indicator"),
# ]

# surfmesh = mlhp.localSimplexCellMesh(triangulation, celldata)

# output0 = mlhp.PVtuOutput(filename="outputs/linear_elasticity_fcm_csg_boundary")
# output1 = mlhp.PVtuOutput(filename="outputs/linear_elasticity_fcm_csg_fcmmesh")

# mlhp.basisOutput(basis, surfmesh, output0, processors)
# mlhp.basisOutput(basis, output=output1, processors=processors)
# # mlhp.basisOutput(basis, mlhp.quadraturePointCellMesh(quadrature, basis), output=mlhp.PVtuOutput(filename="outputs/linear_elasticity_fcm_csg_quadraturepoints"), processors=[])

# # Output boundary surfaces
# surfmesh0 = mlhp.localSimplexCellMesh(intersected0, celldata0)
# surfmesh1 = mlhp.localSimplexCellMesh(intersected1, celldata1)

# surfoutput0 = mlhp.VtuOutput(filename="outputs/linear_elasticity_fcm_csg_boundary0")
# surfoutput1 = mlhp.VtuOutput(filename="outputs/linear_elasticity_fcm_csg_boundary1")

# mlhp.basisOutput(basis, surfmesh0, surfoutput0, processors)
# mlhp.basisOutput(basis, surfmesh1, surfoutput1, processors)

# # Backwards consistency check
# # assert abs(sum([mlhp.norm(diff) for diff in gradient]) - 1.57227628e-04) < 1e-12
