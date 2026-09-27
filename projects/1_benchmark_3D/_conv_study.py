import json
from pathlib import Path

import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
FIXTURE_DIR = DATA_DIR / "elasticity/fixture"

D = 3
GEOMETRY = 4
CASE = 1
DEGREE = 1
REFINEMENT = 1
ALPHA_FCM = 1e-4
RTOL = 1e-8
MAXITER = 8000
NELEMENTS_LIST = [32, 48, 64, 80, 96]
LOG = BASE_DIR / "_conv_study.log"
log = LOG.open("w")


def emit(s):
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()

name = f"{GEOMETRY:09}_abc"
spec = json.loads((FIXTURE_DIR / f"{name}_{CASE}.json").read_text())
surface = mlhp.readStl(str(STL_DIR / spec["stl"]))
kdtree = mlhp.buildKdTree(surface)
domain = mlhp.rayIntersectionDomain(surface, tree=kdtree)
E = spec["material"]["E"]
nu = spec["material"]["nu"]
penalty = 1e5 * E
lo, hi = surface.boundingBox()
lo, hi = np.asarray(lo), np.asarray(hi)
extent = hi - lo


def solve(NELEMENTS):
    h = extent.max() / NELEMENTS
    nelements = [max(int(np.ceil(extent[d] / h)), 1) for d in range(D)]
    box = np.array([n * h for n in nelements])
    origin = (lo - 0.5 * (box - extent) - 1e-9).tolist()
    lengths = (box + 2e-9).tolist()

    grid = mlhp.makeRefinedGrid(nelements, lengths, origin)
    if REFINEMENT:
        grid.refine(mlhp.refineTowardsBoundary(domain, REFINEMENT))
    basis = mlhp.makeHpTrunkSpace(grid, DEGREE, nfields=D)

    kinematics = mlhp.smallStrainKinematics(D)
    material = mlhp.isotropicElasticMaterial(
        mlhp.scalarField(D, E), mlhp.scalarField(D, nu)
    )
    integrand = mlhp.staticDomainIntegrand(
        kinematics, material, mlhp.vectorField(D, [0.0] * D)
    )
    quadrature = mlhp.momentFittingQuadrature(domain, depth=DEGREE, epsilon=ALPHA_FCM)
    matrix = mlhp.allocateSparseMatrix(basis)
    vector = mlhp.allocateRhsVector(matrix)
    mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature)

    for b in spec["boundaries"]:
        grip = mlhp.readStl(str(FIXTURE_DIR / b["stl"]))
        intersected, celldata = mlhp.intersectWithMesh(grip, grid)
        quad = mlhp.simplexQuadrature(intersected, celldata)
        if b["type"] == "dirichlet":
            bc = mlhp.l2BoundaryIntegrand(
                mlhp.vectorField(D, [penalty] * D), mlhp.vectorField(D, b["value"])
            )
            mlhp.integrateOnSurface(basis, bc, [matrix, vector], quad)
        else:
            bc = mlhp.neumannIntegrand(mlhp.vectorField(D, b["value"]))
            mlhp.integrateOnSurface(basis, bc, [vector], quad)

    P = mlhp.additiveSchwarzPreconditioner(matrix, basis)
    dofs, norms = mlhp.cg(matrix, vector, M=P, rtol=RTOL, maxiter=MAXITER, residualNorms=True)
    f = np.asarray(vector)
    compliance = float(np.asarray(dofs) @ f)
    return basis.ndof(), len(norms), norms[-1], float(np.max(np.abs(dofs))), compliance


emit(f"{'N':>5} {'ndof':>10} {'CGit':>6} {'resid':>10} {'max|u|':>12} {'compliance C':>14}")
rows = []
for N in NELEMENTS_LIST:
    ndof, it, res, maxu, C = solve(N)
    rows.append((N, maxu, C))
    emit(f"{N:>5} {ndof:>10} {it:>6} {res:>10.1e} {maxu:>12.5e} {C:>14.6e}")

emit("\n  refinement-to-refinement change (vs previous N):")
emit(f"{'N':>5} {'d max|u|':>12} {'d C':>12}")
for i in range(1, len(rows)):
    du = 100 * (rows[i][1] / rows[i - 1][1] - 1)
    dC = 100 * (rows[i][2] / rows[i - 1][2] - 1)
    emit(f"{rows[i][0]:>5} {du:>11.2f}% {dC:>11.2f}%")
log.close()
