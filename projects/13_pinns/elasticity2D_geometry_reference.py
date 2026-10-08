import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import mlhp.mkl
import numpy as np

from helper import GYROID_Z, SHEET, gyroid_mask, gyroid_threshold

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
DIM = 2
CELLS = np.round(np.arange(1.0, 12.01, 0.5), 1)  # gyroid unit cells per side, continuous
FIELD_CELLS = [1.0, 2.0, 4.0, 8.0]  # settings whose fields are stored for the figure

# discretization, finite cell method on the implicit geometry
DEGREE = 5  # reference, about 1.6 million dofs
NELEMENTS = 320
ALPHA = 1e-8  # finite cell stabilization of the void
STUDY = [(1, 64), (1, 128), (2, 64), (2, 128), (2, 256)]  # (degree, elements per side)
DIRECT_DOFS = 2_000_000  # sparse direct below, preconditioned CG above
RESOLUTION = 256  # evaluation grid (voxel centres) for the displacement error
FIELD_RESOLUTION = 512  # evaluation grid of the stored fields

# physics, plane stress, clamped on the left, loaded downwards on the right edge
E = 1.0
NU = 0.3
TRACTION = [0.0, -1.0]


# --------------------------------------- helper --------------------------------------
def domain(cells):
    """gyroid slice below the threshold, joined to solid face sheets on both edges."""
    k, z = 2 * np.pi * cells, 2 * np.pi * GYROID_Z
    phi = f"sin({k}*x)*cos({k}*y) + sin({k}*y)*{np.cos(z)} + {np.sin(z)}*cos({k}*x)"
    gyroid = mlhp.implicitFunction(DIM, f"{phi} < {gyroid_threshold(cells)}")
    left = mlhp.implicitCube([-1.0, -1.0], [SHEET, 2.0])
    right = mlhp.implicitCube([1.0 - SHEET, -1.0], [2.0, 2.0])
    return mlhp.implicitUnion([gyroid, left, right])


def linear_solve(matrix, vector):
    """Pardiso for small systems, Jacobi-preconditioned CG for large ones (3D)."""
    if len(vector) < DIRECT_DOFS:
        with mlhp.mkl.pardisoFactorize(matrix, symmetric=True) as solve:
            return solve(vector)
    P = mlhp.diagonalPreconditioner(matrix)
    return mlhp.cg(matrix, vector, M=P, rtol=1e-10, maxiter=100000)


def solve(cells, nelements, degree, points=()):
    """finite cell solution: number of dofs, compliance, time of assembly and solve,
    and the displacements (n, 2) at each array of points (n, 2), NaN outside."""
    tic = time.time()
    geometry = domain(cells)
    grid = mlhp.makeGrid([nelements] * DIM, [1.0] * DIM)
    grid = mlhp.makeRefinedGrid(
        mlhp.makeFilteredGrid(grid, domain=geometry, nseedpoints=degree + 2)
    )
    basis = mlhp.makeHpTrunkSpace(grid, degree=degree, nfields=DIM)

    zero = mlhp.vectorField(DIM, [0.0] * DIM)
    dirichlet = mlhp.integrateDirichletDofs(zero, basis, [0])
    matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)

    kinematics = mlhp.smallStrainKinematics(DIM)
    constitutive = mlhp.planeStressMaterial(
        mlhp.scalarField(DIM, E), mlhp.scalarField(DIM, NU)
    )
    integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, zero)
    quadrature = mlhp.spaceTreeQuadrature(geometry, depth=degree + 2, epsilon=ALPHA)
    mlhp.integrateOnDomain(
        basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
    )
    traction = mlhp.neumannIntegrand(mlhp.vectorField(DIM, TRACTION))
    right = mlhp.quadratureOnMeshFaces(grid, [1])
    mlhp.integrateOnSurface(basis, traction, [vector], right, dirichletDofs=dirichlet)

    interior_dofs = linear_solve(matrix, vector)
    elapsed = time.time() - tic
    compliance = float(np.dot(vector.buffer, interior_dofs.buffer))
    dofs = mlhp.inflateDofs(interior_dofs, dirichlet)

    evaluator = mlhp.mechanicalEvaluator(basis, dofs, kinematics, constitutive)
    displacements = []
    for x in points:
        u = np.array(evaluator.displacement(x[:, 0].tolist(), x[:, 1].tolist()))
        u = u.reshape(-1, DIM)
        u[~gyroid_mask(x, cells)] = np.nan
        displacements.append(u)
    return basis.ndof(), compliance, elapsed, displacements


def centres(resolution):
    axis = (np.arange(resolution) + 0.5) / resolution
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, DIM)


# ---------------------------------------- solve --------------------------------------
x_eval, x_field = centres(RESOLUTION), centres(FIELD_RESOLUTION)
results = {"cells": CELLS, "compliance": [], "ndof": [], "time": [], "u": []}
study = {key: [] for key in ["cells", "degree", "elements", "ndof", "compliance", "time", "error"]}
for cells in CELLS:
    points = [x_eval, x_field] if cells in FIELD_CELLS else [x_eval]
    ndof, compliance, elapsed, u = solve(cells, NELEMENTS, DEGREE, points)
    print(f"{cells} cells reference ndof {ndof} compliance {compliance:.8f} time {elapsed:.1f} s")
    for key, value in zip(["compliance", "ndof", "time", "u"], [compliance, ndof, elapsed, u[0]]):
        results[key].append(value)
    if cells in FIELD_CELLS:
        results[f"field_{cells:g}"] = u[1]
    solid = ~np.isnan(u[0][:, 0])
    for degree, nelements in STUDY:
        ndof, c, elapsed, (u_fem,) = solve(cells, nelements, degree, [x_eval])
        error = np.linalg.norm(u_fem[solid] - u[0][solid]) / np.linalg.norm(u[0][solid])
        print(f"  p={degree} {nelements} elements ndof {ndof} time {elapsed:.3f} s error {error:.2e}")
        values = [cells, degree, nelements, ndof, c, elapsed, error]
        for key, value in zip(study, values):
            study[key].append(value)

# -------------------------------------- export ---------------------------------------
results = {key: np.array(value) for key, value in results.items()}
results.update({f"study_{key}": np.array(value) for key, value in study.items()})
results["u"] = results["u"].astype(np.float32)
np.savez(DATA_DIR / "elasticity2D_geometry_reference.npz", **results)
print(f"saved {DATA_DIR / 'elasticity2D_geometry_reference.npz'}")

# ----------------------------------- postprocessing ----------------------------------
fig, axs = plt.subplots(1, len(FIELD_CELLS), figsize=(4 * len(FIELD_CELLS), 4))
for ax, cells in zip(axs, FIELD_CELLS):
    field = results[f"field_{cells:g}"][:, 1].reshape(FIELD_RESOLUTION, FIELD_RESOLUTION)
    ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap="turbo")
    ax.set_title(f"{cells:g} cells")
    ax.axis("off")
plt.show()
