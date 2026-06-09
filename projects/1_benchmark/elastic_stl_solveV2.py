from pathlib import Path

import mlhp
import numpy as np
from pyevtk.hl import imageToVTK

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
RESULTS_DIR = (BASE_DIR / "../../results/abc/elasticity/solution_stl").resolve()

D = 3
name = "000000000_abc"

# ----------------------------------- solver settings ---------------------------------
DEGREE = 1
NELEMENTS = 50
REFINEMENT = 1
ALPHA_FCM = 1e-4
RTOL = 1e-8
MAXITER = 5000

# ----------------------------------- uniaxial tension --------------------------------
# Hardcoded textbook tension to debug the BC path without the fixture pipeline: roller
# (single component) on the three low faces + uniform traction on the +x face. The
# result is constant strain, so the closed form below is exact for linear elements.
E = 210.0
NU = 0.3
TRACTION = 1.0  # tensile traction on the +x face
AXIS = 0  # pull along x (the bar's long axis)
penalty = 1e5 * E

# nodal grid the FE solution is sampled on and compared against (solid bar box)
LENGTHS_BAR = (0.2, 0.05, 0.05)
ORIGIN_BAR = (0.0, 0.0, 0.0)
NCELLS = (40, 10, 10)


# ----------------------------------- helpers -----------------------------------------
def sample_field(field, points, chunk=200000):
    out = np.empty((len(points), field.odim))
    for i in range(0, len(points), chunk):
        p = points[i : i + chunk]
        out[i : i + chunk] = np.array(field(p[:, 0], p[:, 1], p[:, 2])).reshape(
            -1, field.odim
        )
    return out


def rect_grip(corners):  # 4 ccw corners -> two-triangle surface patch
    verts = np.asarray(corners, dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    return mlhp.triangulation(verts, tris)


# ----------------------------------- problem setup -----------------------------------
surface = mlhp.readStl(str(STL_DIR / f"{name}.stl"))
kdtree = mlhp.buildKdTree(surface)
domain = mlhp.rayIntersectionDomain(surface, tree=kdtree)

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
    mlhp.scalarField(D, E), mlhp.scalarField(D, NU)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, material, mlhp.vectorField(D, [0.0] * D)
)
quadrature = mlhp.momentFittingQuadrature(domain, depth=DEGREE, epsilon=ALPHA_FCM)

matrix = mlhp.allocateSparseMatrix(basis)
vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector], quadrature=quadrature)

# ----------------------------------- boundary conditions -----------------------------
# Roller penalty on the three low faces (only the normal component is penalized) plus a
# uniform traction on the high +x face. Grips are axis-aligned rectangles on the bar
# surface, integrated through the same machinery the fixture path uses.
x0, y0, z0 = ORIGIN_BAR
x1, y1, z1 = (ORIGIN_BAR[d] + LENGTHS_BAR[d] for d in range(D))
rollers = [
    (rect_grip([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]), [penalty, 0.0, 0.0]),
    (rect_grip([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]), [0.0, penalty, 0.0]),
    (rect_grip([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]), [0.0, 0.0, penalty]),
]
for grip, weights in rollers:
    intersected, celldata = mlhp.intersectWithMesh(grip, grid)
    quad = mlhp.simplexQuadrature(intersected, celldata)
    bc = mlhp.l2BoundaryIntegrand(
        mlhp.vectorField(D, weights), mlhp.vectorField(D, [0.0] * D)
    )
    mlhp.integrateOnSurface(basis, bc, [matrix, vector], quad)

load = rect_grip([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]])
intersected, celldata = mlhp.intersectWithMesh(load, grid)
quad = mlhp.simplexQuadrature(intersected, celldata)
traction = [0.0] * D
traction[AXIS] = TRACTION
bc = mlhp.neumannIntegrand(mlhp.vectorField(D, traction))
mlhp.integrateOnSurface(basis, bc, [vector], quad)

# -------------------------------------- solve ----------------------------------------
P = mlhp.additiveSchwarzPreconditioner(matrix, basis)
dofs, norms = mlhp.cg(
    matrix, vector, M=P, rtol=RTOL, maxiter=MAXITER, residualNorms=True
)
print(
    f"{name} uniaxial tension: {len(norms)} CG iters, residual {norms[-1]:.2e}, "
    f"max |u| {np.max(np.abs(dofs)):.3e}",
    flush=True,
)

# ----------------------------------- analytical check --------------------------------
nnodes = tuple(n + 1 for n in NCELLS)
spacing = LENGTHS_BAR[0] / NCELLS[0]
axes = [ORIGIN_BAR[d] + spacing * np.arange(nnodes[d]) for d in range(D)]
grids = np.meshgrid(*axes, indexing="ij")
points = np.column_stack([g.ravel() for g in grids])

mech = mlhp.mechanicalEvaluator(basis, dofs, kinematics, material)
displacement = sample_field(mech.displacement, points).reshape(*nnodes, D)

coeff = TRACTION / E
analytical = np.zeros((*nnodes, D))
analytical[..., 0] = coeff * grids[0]
analytical[..., 1] = -NU * coeff * grids[1]
analytical[..., 2] = -NU * coeff * grids[2]
error = displacement - analytical

for c, lbl in enumerate("xyz"):
    rel = np.linalg.norm(error[..., c]) / np.linalg.norm(analytical[..., c])
    print(
        f"\tu_{lbl}: max|err| {np.abs(error[..., c]).max():.3e}, rel L2 {rel:.3e}",
        flush=True,
    )

# ----------------------------------- export ------------------------------------------
indicator = np.full(NCELLS, 255, dtype=np.uint8)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = RESULTS_DIR / f"{name}_V2_voxel"
imageToVTK(
    str(out),
    origin=tuple(float(o) for o in ORIGIN_BAR),
    spacing=(float(spacing),) * D,
    cellData={"indicator": np.ascontiguousarray(indicator)},
    pointData={
        "displacement": tuple(np.ascontiguousarray(displacement[..., i]) for i in range(D)),
        "analytical": tuple(np.ascontiguousarray(analytical[..., i]) for i in range(D)),
        "error": tuple(np.ascontiguousarray(error[..., i]) for i in range(D)),
    },
)
print(f"\t-> {out}.vti", flush=True)

gradient = mlhp.projectGradient(basis, dofs, quadrature)
processors = [
    mlhp.solutionProcessor(D, dofs, "Displacement"),
    mlhp.stressProcessor(gradient, kinematics, material),
    mlhp.vonMisesProcessor(gradient, kinematics, material, "VonMises"),
]
intersected, celldata = mlhp.intersectWithMesh(surface, grid, tree=kdtree)
surfmesh = mlhp.localSimplexCellMesh(intersected, celldata)
output = mlhp.PVtuOutput(filename=str(RESULTS_DIR / f"{name}_V2"))
mlhp.basisOutput(basis, surfmesh, output, processors)
print(f"\t-> {RESULTS_DIR / f'{name}_V2'}.pvtu", flush=True)
