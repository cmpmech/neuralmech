import time
from pathlib import Path

import mlhp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
DIM = 2
PROBLEMS = {  # direction of the traction on the right edge
    "cantilever": [0.0, -1.0],
    "column": [-1.0, -1e-3],  # axial compression with a lateral imperfection
}
BUCKLING = ["column"]  # problems whose newton steps are checked for stability
TARGETS = (10.0 ** np.arange(-4, 2.01, 0.5)).tolist()  # nonlinearities of the study, where reachable
SCAN = np.geomspace(1e-7, 0.3, 80).tolist()  # tractions that locate the targets

# discretization
REFERENCE = [[80, 8], 5, 4]  # elements, degree and refinement depth towards the clamped corners
CHECK = [[80, 8], 4, 3]  # coarser reference, its difference bounds the reference error
NELEMENTS = [40, 4]
DEGREES = [1, 2]  # finite element study on NELEMENTS, no refinement
RESOLUTION = [160, 16]  # evaluation grid
INCREMENT = 1 / 20  # largest load increment of the reference, the study starts at the full load
NEWTON_ITERS = 25
NEWTON_TOL = 1e-9  # residual relative to the larger of the first residual and the load

# physics, plane strain neo-hookean beam, clamped on the left; its strains stay small, the nonlinearity is geometric
LENGTHS = [1.0, 0.1]
E = 1.0
NU = 0.3


# --------------------------------------- helper --------------------------------------
def to_scipy(matrix):
    data = np.asarray(matrix.data_array)
    indices = np.asarray(matrix.indices_array)
    indptr = np.asarray(matrix.indptr_array)
    return sp.csr_matrix((data, indices, indptr), shape=tuple(matrix.shape))


def newton(basis, dirichlet, right, kinematics, law, dofs, load, stability):
    """newton iterations from dofs at the traction vector load; a converged state with a
    negative tangent eigenvalue (unstable, e.g. the straight column beyond buckling) is
    rejected. Returns dofs, converged, iterations."""
    dofs = mlhp.DoubleVector(np.array(dofs).tolist())
    integrand = mlhp.staticDomainIntegrand(kinematics, law, dofs=dofs)
    traction = mlhp.neumannIntegrand(mlhp.vectorField(DIM, load))
    scale = np.linalg.norm(load) * LENGTHS[1]
    for iteration in range(NEWTON_ITERS):
        matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
        vector = mlhp.allocateRhsVector(matrix)
        mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)
        mlhp.integrateOnSurface(basis, traction, [vector], right, dirichletDofs=dirichlet)
        residual = np.asarray(vector)
        norm = np.linalg.norm(residual)
        norm0 = norm if iteration == 0 else norm0
        if not np.isfinite(norm) or norm > 1e6 * max(norm0, scale):
            break
        K = to_scipy(matrix).tocsc()
        if norm <= max(NEWTON_TOL * max(norm0, scale), 1e-15) and iteration > 0:  # round-off floor at tiny loads
            stable = not stability or eigsh(K, k=1, sigma=-1e-12, return_eigenvectors=False)[0] > 0
            return dofs, stable, iteration
        mlhp.add(dofs, mlhp.inflateDofs(mlhp.DoubleVector(spsolve(K, residual)), dirichlet), out=dofs)
    return dofs, False, iteration


def solve(problem, traction, nelements, degree, depth, increment=1.0, linear=False):
    """incremental loading with steps of at most increment, halved when newton fails or
    ends unstable and doubled (up to increment) when it succeeds; returns the
    displacement on the evaluation grid, time, dofs, load steps and newton iterations."""
    tic = time.time()
    grid = mlhp.makeRefinedGrid(nelements, LENGTHS)
    if depth:
        corners = [[0.0, 0.0], [0.0, LENGTHS[1]]]
        spheres = [mlhp.implicitSphere(c, 1e-9) for c in corners]
        grid.refine(mlhp.refinementOr([mlhp.refineTowardsBoundary(s, depth) for s in spheres]))
    basis = mlhp.makeHpTrunkSpace(grid, degree=degree, nfields=DIM)
    dirichlet = mlhp.integrateDirichletDofs(mlhp.vectorField(DIM, [0.0] * DIM), basis, [0])
    right = mlhp.quadratureOnMeshFaces(grid, [1])
    youngs_modulus, poissons_ratio = mlhp.scalarField(DIM, E), mlhp.scalarField(DIM, NU)
    if linear:
        kinematics, material = mlhp.smallStrainKinematics(DIM), mlhp.planeStrainMaterial(youngs_modulus, poissons_ratio)
    else:
        kinematics, material = mlhp.greenLagrangeStrain(DIM), mlhp.neoHookeanMaterial(youngs_modulus, poissons_ratio)

    dofs = mlhp.DoubleVector(basis.ndof(), 0.0)
    done, step, steps, iterations = 0.0, increment, 0, 0
    while done < 1.0:
        trial = min(done + step, 1.0)
        load = [trial * traction * d for d in PROBLEMS[problem]]
        stability = problem in BUCKLING and not linear
        trial_dofs, converged, count = newton(basis, dirichlet, right, kinematics, material, dofs, load, stability)
        iterations += count
        if converged:
            dofs, done, step, steps = trial_dofs, trial, min(2 * step, increment), steps + 1
        else:
            step /= 2
            if step < 1e-5:
                raise RuntimeError(f"{problem}: newton does not converge at traction {traction}")
    toc = time.time()

    displacement = mlhp.vectorEvaluator(basis, dofs)
    u = np.array(displacement(*coordinates)).reshape(*x_eval.shape)
    return u, toc - tic, basis.ndof(), steps, iterations


def mirror(u):
    """mirror image about the beam axis, the other buckling direction."""
    return np.stack([u[:, ::-1, 0], -u[:, ::-1, 1]], axis=-1)


def difference(u, v):
    return np.sqrt(np.sum((u - v) ** 2) / np.sum(v**2))


def nonlinearity(u, u_linear):
    """relative difference to the linear solution, normalized by the smaller norm so
    that it grows without bound for stiffening and softening responses alike."""
    return np.sqrt(np.sum((u - u_linear) ** 2) / min(np.sum(u**2), np.sum(u_linear**2)))


# ---------------------------------------- solve --------------------------------------
axes = [(np.arange(n) + 0.5) / n * length for n, length in zip(RESOLUTION, LENGTHS)]
x_eval = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
coordinates = [x_eval[..., 0].ravel().tolist(), x_eval[..., 1].ravel().tolist()]

for problem in PROBLEMS:
    # quadratic elements locate the tractions of the target nonlinearities by bisection
    def scan_nonlinearity(traction):
        u, *_ = solve(problem, traction, NELEMENTS, 2, 0)
        u_linear, *_ = solve(problem, traction, NELEMENTS, 2, 0, linear=True)
        return nonlinearity(u, u_linear)

    scan = []
    for traction in SCAN:
        try:
            scan.append(scan_nonlinearity(traction))
        except RuntimeError:
            break
        # the column passes its largest nonlinearity; below the smallest target it is round-off
        if len(scan) > 1 and scan[-2] > TARGETS[0] and scan[-1] < scan[-2]:
            scan.pop()
            break
    targets, tractions = [target for target in TARGETS if scan[0] < target <= scan[-1]], []
    for target in targets:
        i = np.searchsorted(scan, target)
        lower, upper = np.log(SCAN[i - 1]), np.log(SCAN[i])
        for _ in range(12):
            middle = 0.5 * (lower + upper)
            lower, upper = (middle, upper) if scan_nonlinearity(np.exp(middle)) < target else (lower, middle)
        tractions.append(float(np.exp(0.5 * (lower + upper))))
    print(f"{problem}: scan up to nonlinearity {scan[-1]:.2e}, tractions {[f'{t:.4e}' for t in tractions]}")

    results = {"traction": [], "target": [], "x": x_eval[..., 0], "y": x_eval[..., 1]}
    for key in ["nonlinearity", "reference_error", "reference_time", "reference_steps", "reference_iterations"]:
        results[key] = []
    for degree in DEGREES:
        for key in ["error", "time", "ndof", "steps", "iterations"]:
            results[f"p{degree}_{key}"] = []

    for target, traction in zip(targets, tractions):
        try:
            u, elapsed, ndof, steps, iterations = solve(problem, traction, *REFERENCE, INCREMENT)
            u_check, *_ = solve(problem, traction, *CHECK, INCREMENT)
        except RuntimeError as error:
            print(f"{error}, the study ends here")
            break
        results["traction"].append(traction)
        results["target"].append(target)
        u_linear, *_ = solve(problem, traction, *REFERENCE, linear=True)
        values = [nonlinearity(u, u_linear), min(difference(u_check, u), difference(mirror(u_check), u)), elapsed, steps, iterations]
        for key, value in zip(["nonlinearity", "reference_error", "reference_time", "reference_steps", "reference_iterations"], values):
            results[key].append(value)
        results[f"ux_{traction}"], results[f"uy_{traction}"] = u[..., 0], u[..., 1]
        print(
            f"{problem} traction {traction:.3e}: nonlinearity {values[0]:.3e}, reference {ndof} dofs, "
            f"{steps} steps, {iterations} newton iterations, {elapsed:.1f} s, "
            f"difference to p={CHECK[1]} depth {CHECK[2]} {values[1]:.2e}"
        )
        for degree in DEGREES:
            u_fem, elapsed, ndof, steps, iterations = solve(problem, traction, NELEMENTS, degree, 0)
            values = [min(difference(u_fem, u), difference(mirror(u_fem), u)), elapsed, ndof, steps, iterations]
            for key, value in zip(["error", "time", "ndof", "steps", "iterations"], values):
                results[f"p{degree}_{key}"].append(value)
            print(
                f"    p={degree}: error {values[0]:.2e}, {ndof} dofs, {steps} steps, "
                f"{iterations} newton iterations, {elapsed:.2f} s"
            )

# -------------------------------------- export ---------------------------------------
    np.savez(DATA_DIR / f"elasticity2D_nonlinear_{problem}_reference.npz", **results)
    print(f"saved {DATA_DIR / f'elasticity2D_nonlinear_{problem}_reference.npz'}")
