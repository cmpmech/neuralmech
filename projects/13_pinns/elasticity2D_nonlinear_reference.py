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
TRACTIONS = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
]  # sets the nonlinearity, newton fails at 0.5

# discretization
DEGREE = 4  # reference
NELEMENTS = 40
DEPTH = 6
COARSE_ELEMENTS = 5  # finite elements with the error of the network in the linear case
COARSE_DEGREE = 2
STEPS = 5  # load steps
RESOLUTION = 160  # evaluation grid

# physics
E = 1.0
NU = 0.3
HOLES = 1


# --------------------------------------- helper --------------------------------------
def solve(traction, nelements, degree, depth, nonlinear=True):
    tic = time.time()
    centers = (np.arange(nelements) + 0.5) / nelements
    centers = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=-1)
    base_grid = mlhp.makeGrid([nelements] * DIM, [1.0] * DIM, [0.0] * DIM)
    mask = (~in_holes(centers, HOLES)).ravel("C").tolist()
    grid = mlhp.makeRefinedGrid(mlhp.makeFilteredGrid(base_grid, mask=mask))
    if depth:
        corners = [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.4, 0.4],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.6, 0.6],
        ]
        spheres = [mlhp.implicitSphere(c, 1e-9) for c in corners]
        grid.refine(
            mlhp.refinementOr([mlhp.refineTowardsBoundary(s, depth) for s in spheres])
        )
    basis = mlhp.makeHpTrunkSpace(grid, degree=degree, nfields=DIM)

    dirichlet = mlhp.integrateDirichletDofs(
        mlhp.vectorField(DIM, [0.0] * DIM), basis, [0]
    )
    youngs_modulus, poissons_ratio = mlhp.scalarField(DIM, E), mlhp.scalarField(DIM, NU)
    if nonlinear:
        kinematics = mlhp.greenLagrangeStrain(DIM)
        material = mlhp.neoHookeanMaterial(youngs_modulus, poissons_ratio)
    else:
        kinematics = mlhp.smallStrainKinematics(DIM)
        material = mlhp.planeStrainMaterial(youngs_modulus, poissons_ratio)
    dofs = mlhp.DoubleVector(basis.ndof(), 0.0)
    integrand = mlhp.staticDomainIntegrand(kinematics, material, dofs=dofs)
    right = mlhp.quadratureOnMeshFaces(grid, [1])

    steps = STEPS if nonlinear else 1
    for step in range(steps):
        load = mlhp.vectorField(DIM, [traction * (step + 1) / steps, 0.0])
        for iteration in range(50):
            matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
            vector = mlhp.allocateRhsVector(matrix)
            mlhp.integrateOnDomain(
                basis, integrand, [matrix, vector], dirichletDofs=dirichlet
            )
            mlhp.integrateOnSurface(
                basis,
                mlhp.neumannIntegrand(load),
                [vector],
                right,
                dirichletDofs=dirichlet,
            )
            norm = mlhp.norm(vector)
            if iteration == 0:
                norm0 = norm
            if norm <= max(1e-9 * norm0, 1e-12):
                break
            P = mlhp.diagonalPreconditioner(matrix)
            increment = mlhp.cg(matrix, vector, rtol=1e-12, M=P, maxiter=100000)
            mlhp.add(dofs, mlhp.inflateDofs(increment, dirichlet), out=dofs)
        else:
            raise RuntimeError(f"newton did not converge for traction {traction}")
    toc = time.time()

    displacement = mlhp.vectorEvaluator(basis, dofs)
    coordinates = [x_eval[..., 0].ravel().tolist(), x_eval[..., 1].ravel().tolist()]
    u = np.array(displacement(*coordinates)).reshape(*x_eval.shape)
    u[in_holes(x_eval, HOLES)] = np.nan
    return u, toc - tic


# ---------------------------------------- solve --------------------------------------
centers = (np.arange(RESOLUTION) + 0.5) / RESOLUTION
x_eval = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=-1)

results = {"traction": np.array(TRACTIONS), "x": x_eval[..., 0], "y": x_eval[..., 1]}
for key in ["nonlinearity", "error", "time"]:
    results[key] = []
for traction in TRACTIONS:
    u, _ = solve(traction, NELEMENTS, DEGREE, DEPTH)
    u_linear, _ = solve(traction, NELEMENTS, DEGREE, DEPTH, nonlinear=False)
    u_coarse, elapsed = solve(traction, COARSE_ELEMENTS, COARSE_DEGREE, 0)

    norm = np.sqrt(np.nansum(u**2))
    nonlinearity = np.sqrt(np.nansum((u - u_linear) ** 2)) / norm
    error = np.sqrt(np.nansum((u - u_coarse) ** 2)) / norm
    print(
        f"traction {traction} nonlinearity {nonlinearity:.2e} "
        f"coarse error {error:.2e} time {elapsed:.2f} s"
    )
    results["nonlinearity"].append(nonlinearity)
    results["error"].append(error)
    results["time"].append(elapsed)
    results[f"ux_{traction}"], results[f"uy_{traction}"] = u[..., 0], u[..., 1]

# -------------------------------------- export ---------------------------------------
np.savez(DATA_DIR / "elasticity2D_nonlinear_reference.npz", **results)
print(f"saved {DATA_DIR / 'elasticity2D_nonlinear_reference.npz'}")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
traction = TRACTIONS[-1]
ax.pcolormesh(
    results["x"] + np.nan_to_num(results[f"ux_{traction}"]),
    results["y"] + np.nan_to_num(results[f"uy_{traction}"]),
    results[f"ux_{traction}"],
    cmap="turbo",
)
ax.set_aspect("equal")
plt.show()
