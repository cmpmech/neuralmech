import argparse
import itertools
import json
import time
from pathlib import Path

import mlhp
import numpy as np
from pyevtk.hl import imageToVTK
from scipy import ndimage
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
FIXTURE_DIR = DATA_DIR / "elasticity/fixture"

D = 3

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--case", type=int, default=1)
args = parser.parse_args()

# ----------------------------------- solver settings ---------------------------------
DEGREE = 1
NELEMENTS = 120  # 80
REFINEMENT = 0
ALPHA_FCM = 1e-5
RTOL = 1e-8  # CG
MAXITER = 5000  # CG

POSTPROCESSING = True

name = f"{args.geometry:09}_abc"

# ----------------------------------- problem setup -----------------------------------
spec = json.loads((FIXTURE_DIR / f"{name}_{args.case}.json").read_text())
surface = mlhp.readStl(str(STL_DIR / spec["stl"]))
kdtree = mlhp.buildKdTree(surface)
domain = mlhp.rayIntersectionDomain(surface, tree=kdtree)

E = spec["material"]["E"]
nu = spec["material"]["nu"]
penalty = 1e5 * E

lo, hi = surface.boundingBox()  # derive the domain box from the STL (single source)
lo, hi = np.asarray(lo), np.asarray(hi)
extent = hi - lo
h = extent.max() / NELEMENTS  # target (cubic) element size
nelements = [max(int(np.ceil(extent[d] / h)), 1) for d in range(D)]
box = np.array([n * h for n in nelements])  # cubic-cell box, >= extent on every axis
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

# ----------------------------------- boundary conditions -----------------------------
# Each boundary is a standalone grip STL; integrate its penalty fix or traction directly.
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

# -------------------------------------- solve ----------------------------------------
tic = time.time()
P = mlhp.additiveSchwarzPreconditioner(matrix, basis)
dofs, norms = mlhp.cg(
    matrix, vector, M=P, rtol=RTOL, maxiter=MAXITER, residualNorms=True
)
print(
    f"{name} case {args.case} ({spec['label']}): "
    f"{len(norms)} CG iters, residual {norms[-1]:.2e}, "
    f"max |u| {np.max(np.abs(dofs)):.3e}",
    flush=True,
)
print(f"elapsed solve time: {time.time() - tic:.2f} s")
