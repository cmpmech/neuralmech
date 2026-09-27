import time
from pathlib import Path

import mlhp
import numpy as np

from helper import corner_plot_points

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
DIMENSIONS = [1, 2, 3, 4, 5, 6]  # above 3D build mlhp with -DMLHP_DIMENSIONS=6

# discretization
MAX_DEPTH = 20  # refinement levels towards the corner, polynomial degree depth + 1
TIME_LIMIT = 20.0  # seconds, stop refining a dimension after the first slower solve
PLOT_DIMENSIONS = [1, 2, 3]  # fields of the finest solution for plotting
PLOT_RESOLUTION = 100

# physics
GAMMA = 0.65  # 1D only, exponent of x**GAMMA - GAMMA * x


# --------------------------------------- helper --------------------------------------
def analytical_fields(D):
    if D == 1:
        solution = mlhp.scalarField(D, f"x**{GAMMA} - {GAMMA} * x")
        derivatives = mlhp.vectorField(D, f"[{GAMMA} * x**{GAMMA - 1} - {GAMMA}]")
        source = mlhp.scalarField(D, f"{-GAMMA * (GAMMA - 1)} * x**{GAMMA - 2}")
        return solution, derivatives, source
    r2 = "(" + " + ".join(f"xyz[{i}]**2" for i in range(D)) + ")"
    solution = mlhp.scalarField(D, f"{r2}**0.25")
    gradient = ", ".join(f"0.5 * xyz[{i}] / {r2}**0.75" for i in range(D))
    derivatives = mlhp.vectorField(D, f"[{gradient}]")
    source = mlhp.scalarField(D, f"{(3 - 2 * D) / 4} * {r2}**-0.75")
    return solution, derivatives, source


def solve(D, depth):
    solution, derivatives, source = analytical_fields(D)

    tic = time.time()
    grid = mlhp.makeRefinedGrid([2] * D, [1.0] * D)
    grid.refine(mlhp.refineTowardsBoundary(mlhp.implicitSphere([0.0] * D, 0.0), depth))
    basis = mlhp.makeHpTensorSpace(grid, mlhp.UniformGrading(depth + 1))

    faces = [2 * axis + 1 for axis in range(D)] if D > 1 else [0]
    dirichlet = mlhp.integrateDirichletDofs(solution, basis, faces)

    matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)
    integrand = mlhp.poissonIntegrand(mlhp.scalarField(D, 1.0), source)
    mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

    P = mlhp.diagonalPreconditioner(matrix)
    interior_dofs = mlhp.cg(matrix, vector, rtol=1e-10, M=P, maxiter=100000)
    dofs = mlhp.inflateDofs(interior_dofs, dirichlet)
    toc = time.time()

    integrals = mlhp.makeScalars(3)
    mlhp.integrateOnDomain(
        basis,
        mlhp.energyErrorIntegrand(dofs, derivatives),
        integrals,
        orderDeterminor=mlhp.relativeQuadratureOrder(D, 3),
    )
    error = np.sqrt(integrals[2].get() / integrals[1].get())
    return len(interior_dofs), error, toc - tic, mlhp.scalarEvaluator(basis, dofs)


# --------------------------------------- solve ---------------------------------------
results = {"dim": [], "depth": [], "ndof": [], "error": [], "time": []}
fields = {}
for D in DIMENSIONS:
    for depth in range(MAX_DEPTH + 1):
        ndof, error, elapsed, evaluator = solve(D, depth)
        print(f"{D}D depth {depth} ndof {ndof} error {error:.2e} time {elapsed:.2f} s")
        for key, value in zip(results, [D, depth, ndof, error, elapsed]):
            results[key].append(value)
        if D in PLOT_DIMENSIONS:
            x_plot = corner_plot_points(D, PLOT_RESOLUTION)
            fields[f"u_{D}D"] = np.array(evaluator(*x_plot.T.tolist()))
        if elapsed > TIME_LIMIT:
            break

# -------------------------------------- export ---------------------------------------
np.savez(DATA_DIR / "poisson_corner_reference.npz", **results, **fields)
print(f"saved {DATA_DIR / 'poisson_corner_reference.npz'}")
