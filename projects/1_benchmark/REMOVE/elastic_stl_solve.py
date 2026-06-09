import argparse
import json
from pathlib import Path

import mlhp
import numpy as np
from pyevtk.hl import imageToVTK
from scipy import ndimage

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
VOXEL_DIR = DATA_DIR / "geometry/voxel"
SOLUTION_DIR = DATA_DIR / "elasticity/solution_stl"
FIXTURE_DIR = DATA_DIR / "elasticity/fixture"
RESULTS_SOLUTION_DIR = (
    BASE_DIR / "../../results/abc/elasticity/solution_stl"
).resolve()
RESULTS_FIXTURE_DIR = (BASE_DIR / "../../results/abc/elasticity/fixture").resolve()

D = 3

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--case", type=int, default=1)
args = parser.parse_args()

# ----------------------------------- solver settings ---------------------------------
DEGREE = 1
NELEMENTS = 30
REFINEMENT = 1
ALPHA_FCM = 1e-4
RTOL = 1e-8
MAXITER = 5000

POSTPROCESSING = True

name = f"{args.geometry:09}_abc"


# ----------------------------------- helpers -----------------------------------------
def sample_field(field, points, chunk=200000):
    out = np.empty((len(points), field.odim))
    for i in range(0, len(points), chunk):
        p = points[i : i + chunk]
        out[i : i + chunk] = np.array(field(p[:, 0], p[:, 1], p[:, 2])).reshape(
            -1, field.odim
        )
    return out


def rasterize(grip, voxgrid, ncells):  # voxel cells the grip surface passes through
    _, celldata = mlhp.intersectWithMesh(grip, voxgrid)
    cut = np.zeros(ncells, dtype=bool)
    cut[np.unravel_index(np.array(celldata.meshSupport()), ncells, order="C")] = True
    return cut


def to_float16(arr, label):  # storage cast; warn if it overflows or flushes data to 0
    a16 = arr.astype(np.float16)
    scale = np.abs(arr).max()
    overflow = np.isinf(a16) & np.isfinite(arr)  # > 65504 -> inf
    underflow = (a16 == 0) & (arr != 0)  # nonzero value rounded to 0
    if overflow.any():
        print(
            f"\tWARNING: {label} float16 overflow: {int(overflow.sum())} values "
            f"> 65504 became inf (max |x| {scale:.3e})",
            flush=True,
        )
    if underflow.any():
        worst = np.abs(arr[underflow]).max()
        print(
            f"\tWARNING: {label} float16 underflow: {int(underflow.sum())} nonzero "
            f"values flushed to 0 (largest {worst:.3e}, {worst / scale:.0e} of peak)",
            flush=True,
        )
    return a16


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
P = mlhp.additiveSchwarzPreconditioner(matrix, basis)
dofs, norms = mlhp.cg(
    matrix, vector, M=P, rtol=RTOL, maxiter=MAXITER, residualNorms=True
)
print(
    f"{name} case {args.case} ({spec['label']}): "
    f"{len(norms)} CG iters, residual {norms[-1]:.2e}",
    flush=True,
)

# ----------------------------------- surface vtu -------------------------------------
if POSTPROCESSING:
    gradient = mlhp.projectGradient(basis, dofs, quadrature)
    processors = [
        mlhp.solutionProcessor(D, dofs, "Displacement"),
        mlhp.stressProcessor(gradient, kinematics, material),
        mlhp.vonMisesProcessor(gradient, kinematics, material, "VonMises"),
    ]
    intersected, celldata = mlhp.intersectWithMesh(surface, grid, tree=kdtree)
    surfmesh = mlhp.localSimplexCellMesh(intersected, celldata)
    RESULTS_SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    output = mlhp.PVtuOutput(filename=str(RESULTS_SOLUTION_DIR / f"{name}_{args.case}"))
    mlhp.basisOutput(basis, surfmesh, output, processors)

# ----------------------------------- voxel grid --------------------------------------
vox = np.load(VOXEL_DIR / f"{name}.npz")
indicator = vox["indicator"]
ncells = indicator.shape
lengths_v = np.array([float(vox["Lx"]), float(vox["Ly"]), float(vox["Lz"])])
spacing = lengths_v[0] / ncells[0]  # cubic voxels
origin_v = lo - 0.5 * (lengths_v - (hi - lo))  # matches voxelization.py

axes = [origin_v[d] + spacing * (np.arange(ncells[d]) + 0.5) for d in range(D)]
points = np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])
mask = indicator.ravel() >= 128  # solid voxels

# ----------------------------------- voxel solution ----------------------------------
mech = mlhp.mechanicalEvaluator(basis, dofs, kinematics, material)
disp = sample_field(mech.displacement, points[mask])
stress = sample_field(mech.stress, points[mask])  # (n, 9), row-major 3x3 Cauchy tensor

disp_full = np.zeros((points.shape[0], 3))
stress_full = np.zeros((points.shape[0], 6))
disp_full[mask] = disp
# store the 6 unique components of the symmetric tensor: xx, yy, zz, xy, yz, xz
stress_full[mask] = stress[:, [0, 4, 8, 1, 5, 2]]

displacement = disp_full.reshape(*ncells, 3)
stress_voigt = stress_full.reshape(*ncells, 6)
print(f"\tvoxelized {mask.sum()}/{mask.size} solid voxels", flush=True)

# displacement and stress to separate files so a displacement-only target needn't
# carry (or load) the heavier 6-component stress. indicator is omitted (it's geometry,
# in data/abc/geometry/voxel/<name>.npz); each file keeps the grid metadata.
disp_dir = SOLUTION_DIR / "displacement"
stress_dir = SOLUTION_DIR / "stress"
disp_dir.mkdir(parents=True, exist_ok=True)
stress_dir.mkdir(parents=True, exist_ok=True)
meta = dict(
    origin=origin_v, spacing=spacing, Lx=lengths_v[0], Ly=lengths_v[1], Lz=lengths_v[2]
)
disp_out = disp_dir / f"{name}_{args.case}.npz"
stress_out = stress_dir / f"{name}_{args.case}.npz"
np.savez_compressed(
    disp_out, displacement=to_float16(displacement, "displacement"), **meta
)
np.savez_compressed(
    stress_out, stress=to_float16(stress_voigt, "stress"), **meta
)  # xx,yy,zz,xy,yz,xz
print(f"\t-> {disp_out}", flush=True)
print(f"\t-> {stress_out}", flush=True)

if POSTPROCESSING:
    RESULTS_SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_SOLUTION_DIR / f"{name}_{args.case}_voxel"
    cell_data = {
        "indicator": np.ascontiguousarray(indicator),
        "displacement": tuple(
            np.ascontiguousarray(displacement[..., i]) for i in range(D)
        ),
    }
    imageToVTK(
        str(out),
        origin=tuple(
            float(x) for x in origin_v
        ),  # plain floats: np.float64 str() breaks the .vti
        spacing=(float(spacing),) * D,
        cellData=cell_data,
    )
    print(f"\t-> {out}.vti", flush=True)

# ----------------------------------- voxel fixture -----------------------------------
# Rasterize each grip onto the voxel grid and stamp its BC onto the outer-hull voxels
# (the outside shell the grip passes through), so the surface is the unique hull/solid face.
solid = indicator >= 128
structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
hull = ndimage.binary_dilation(solid, structure=structure) & ~solid  # outside shell
voxgrid = mlhp.makeRefinedGrid(
    list(ncells), [float(x) for x in lengths_v], [float(x) for x in origin_v]
)
dirichlet_mask = np.zeros((*ncells, D), dtype=np.uint8)
dirichlet_value = np.zeros((*ncells, D))
neumann = np.zeros((*ncells, D))
for b in spec["boundaries"]:
    grip = mlhp.readStl(str(FIXTURE_DIR / b["stl"]))
    tag = rasterize(grip, voxgrid, ncells) & hull  # hull cells over the grip
    if b["type"] == "dirichlet":
        dirichlet_mask[tag] = 1
        dirichlet_value[tag] = b["value"]
    else:
        neumann[tag] = b["value"]
print(
    f"\tfixture {int((dirichlet_mask[..., 0] > 0).sum())} dirichlet,"
    f" {int(np.any(neumann != 0, axis=-1).sum())} neumann hull voxels",
    flush=True,
)

voxel_dir = FIXTURE_DIR / "boundary_voxel"
voxel_dir.mkdir(parents=True, exist_ok=True)
out = voxel_dir / f"{name}_{args.case}.npz"
np.savez_compressed(
    out,
    indicator=indicator,
    dirichlet_mask=dirichlet_mask,
    dirichlet_value=dirichlet_value.astype(np.float32),
    neumann=neumann.astype(np.float32),
    origin=origin_v,
    spacing=spacing,
    Lx=lengths_v[0],
    Ly=lengths_v[1],
    Lz=lengths_v[2],
)
print(f"\t-> {out}", flush=True)

if POSTPROCESSING:
    RESULTS_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_FIXTURE_DIR / f"{name}_{args.case}"
    cell_data = {
        "dirichlet": tuple(
            np.ascontiguousarray(dirichlet_mask[..., i]) for i in range(D)
        ),
        "neumann": tuple(np.ascontiguousarray(neumann[..., i]) for i in range(D)),
    }
    imageToVTK(
        str(out),
        origin=tuple(
            float(x) for x in origin_v
        ),  # plain floats: np.float64 str() breaks the .vti
        spacing=(float(spacing),) * D,
        cellData=cell_data,
    )
    print(f"\t-> {out}.vti", flush=True)
