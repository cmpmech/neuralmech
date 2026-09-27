import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

from helper import in_holes

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
DIM = 2
HOLES = [1, 2, 4, 8]  # holes per side of the perforated plate

# discretization
DEGREE = 6  # reference
NELEMENTS = 40  # reference, multiple of 5 * holes, so the hole edges lie on elements
DEPTH = 10  # reference refinement levels towards the singular corners
ELEMENTS = [5, 10, 20, 40, 80, 160]  # finite element study, uniform meshes
DEGREES = [1, 2]
RESOLUTION = 320  # evaluation grid, multiple of 5 * holes

# physics
E = 1.0
NU = 0.3
TRACTION = 1.0


# --------------------------------------- helper --------------------------------------
def singular_corners(holes):
    corners = [[0.0, 0.0], [0.0, 1.0]]  # clamped corners
    for i in range(holes):
        for j in range(holes):
            for a in [0.4, 0.6]:
                for b in [0.4, 0.6]:
                    corners.append([(i + a) / holes, (j + b) / holes])
    return corners


def solve(holes, nelements, degree, depth=0):
    tic = time.time()
    centers = (np.arange(nelements) + 0.5) / nelements
    centers = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=-1)
    base_grid = mlhp.makeGrid([nelements] * DIM, [1.0] * DIM, [0.0] * DIM)
    mask = (~in_holes(centers, holes)).ravel("C").tolist()
    grid = mlhp.makeRefinedGrid(mlhp.makeFilteredGrid(base_grid, mask=mask))
    if depth:
        # a sphere of radius 0 does not refine at corners inside the plate
        spheres = [mlhp.implicitSphere(c, 1e-9) for c in singular_corners(holes)]
        grid.refine(
            mlhp.refinementOr([mlhp.refineTowardsBoundary(s, depth) for s in spheres])
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
    mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)
    traction = mlhp.neumannIntegrand(mlhp.vectorField(DIM, [TRACTION, 0.0]))
    right = mlhp.quadratureOnMeshFaces(grid, [1])
    mlhp.integrateOnSurface(basis, traction, [vector], right, dirichletDofs=dirichlet)

    P = mlhp.diagonalPreconditioner(matrix)
    interior_dofs = mlhp.cg(matrix, vector, rtol=1e-12, M=P, maxiter=100000)
    toc = time.time()
    compliance = np.dot(vector.array, interior_dofs.array)
    dofs = mlhp.inflateDofs(interior_dofs, dirichlet)

    evaluator = mlhp.mechanicalEvaluator(basis, dofs, kinematics, constitutive)
    in_hole = in_holes(x_eval, holes)
    coordinates = [x_eval[..., 0].ravel().tolist(), x_eval[..., 1].ravel().tolist()]
    u = np.array(evaluator.displacement(*coordinates)).reshape(-1, DIM)
    gradient = np.array(evaluator.displacementGradient(*coordinates)).reshape(-1, 4)
    fields = {
        "ux": u[:, 0].reshape(in_hole.shape),
        "uy": u[:, 1].reshape(in_hole.shape),
        "exx": gradient[:, 0].reshape(in_hole.shape),
    }
    fields = {key: np.where(in_hole, np.nan, value) for key, value in fields.items()}
    return basis.ndof(), compliance, toc - tic, fields


# ---------------------------------------- solve --------------------------------------
centers = (np.arange(RESOLUTION) + 0.5) / RESOLUTION
x_eval = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=-1)

results = {"holes": HOLES, "x": x_eval[..., 0], "y": x_eval[..., 1], "compliance": []}
study = {
    key: [] for key in ["holes", "elements", "degree", "ndof", "compliance", "time"]
}
for holes in HOLES:
    ndof, compliance, elapsed, fields = solve(holes, NELEMENTS, DEGREE, DEPTH)
    print(f"{holes} holes reference ndof {ndof} compliance {compliance:.8f}")
    results["compliance"].append(compliance)
    results.update({f"{key}_{holes}": value for key, value in fields.items()})

    for degree in DEGREES:
        for nelements in ELEMENTS:
            if nelements % (5 * holes):
                continue
            ndof, c, elapsed, fields = solve(holes, nelements, degree)
            error = np.sqrt((compliance - c) / compliance)
            print(f"{holes} holes {nelements} elements p={degree} error {error:.2e}")
            for key, value in zip(study, [holes, nelements, degree, ndof, c, elapsed]):
                study[key].append(value)
            name = f"{holes}_{nelements}_{degree}"
            results.update({f"{key}_{name}": value for key, value in fields.items()})

# -------------------------------------- export ---------------------------------------
results.update({f"study_{key}": np.array(value) for key, value in study.items()})
np.savez(DATA_DIR / "elasticity2D_reference.npz", **results)
print(f"saved {DATA_DIR / 'elasticity2D_reference.npz'}")

# ----------------------------------- postprocessing ----------------------------------
fig, axs = plt.subplots(1, len(HOLES), figsize=(4 * len(HOLES), 4))
for ax, holes in zip(axs, HOLES):
    ax.pcolormesh(results["x"], results["y"], results[f"ux_{holes}"], cmap="turbo")
    ax.set_aspect("equal")
plt.show()
