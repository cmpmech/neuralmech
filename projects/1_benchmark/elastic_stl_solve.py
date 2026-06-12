import argparse
import itertools
import json
from pathlib import Path

import mlhp
import numpy as np
from pyevtk.hl import imageToVTK
from scipy import ndimage
from scipy.spatial import cKDTree

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
DEGREE = 1  # 1
NELEMENTS = 10  # 30  # 50  # 30
REFINEMENT = 1  # 2
ALPHA_FCM = 1e-5  # 1e-4
RTOL = 1e-8
MAXITER = 5000

POSTPROCESSING = True

name = f"{args.geometry:09}_abc"

# corner-offset table: local node l sits at the grid vertex (i + l>>2&1, j + l>>1&1, k + l&1)
OFFSET = np.array([[(l >> 2) & 1, (l >> 1) & 1, l & 1] for l in range(8)])


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
    support = np.asarray(
        celldata.meshSupport(), dtype=np.int64
    )  # empty -> float, guard
    if support.size:
        cut[np.unravel_index(support, ncells, order="C")] = True
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


def shift_or(cells, ncells):  # promote a cell-mask to its corner nodes (2^D OR)
    nodes = np.zeros(tuple(n + 1 for n in ncells), dtype=bool)
    for off in itertools.product((0, 1), repeat=D):
        sl = tuple(slice(o, o + ncells[d]) for d, o in enumerate(off))
        nodes[sl] |= cells
    return nodes


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
    f"{len(norms)} CG iters, residual {norms[-1]:.2e}, "
    f"max |u| {np.max(np.abs(dofs)):.3e}",
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

    # element mesh -> "<name>_mesh.pvtu": the FE grid with cell edges (gridCellMesh
    # defaults to Volumes|Edges), so the elements are visible alongside the solution.
    meshcells = mlhp.gridCellMesh([DEGREE + 1] * D)
    meshoutput = mlhp.PVtuOutput(
        filename=str(RESULTS_SOLUTION_DIR / f"{name}_{args.case}_mesh")
    )
    mlhp.basisOutput(basis, meshcells, meshoutput, processors)

# ----------------------------------- nodal grid --------------------------------------
# Staggered export: displacement lives at voxel CORNERS (the (N+1)^3 nodal grid), the
# natural FE node positions. The 8 corner displacements of a voxel define a trilinear
# field whose gradient at the cell center is the exact FE strain, so cell-centered
# stress is analytically recoverable downstream and need not be stored here.
vox = np.load(VOXEL_DIR / f"{name}.npz")
indicator = vox["indicator"]
ncells = indicator.shape
lengths_v = np.array([float(vox["Lx"]), float(vox["Ly"]), float(vox["Lz"])])
spacing = lengths_v[0] / ncells[0]  # cubic voxels
origin_v = lo - 0.5 * (lengths_v - (hi - lo))  # node 0 sits exactly at origin_v

nnodes = tuple(n + 1 for n in ncells)
axes = [origin_v[d] + spacing * np.arange(nnodes[d]) for d in range(D)]  # corners
points = np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])

solid = indicator >= 128
node_solid = shift_or(solid, ncells)  # node is solid iff any adjacent cell is solid
node_mask = node_solid.ravel()

# ----------------------------------- nodal solution ----------------------------------
mech = mlhp.mechanicalEvaluator(basis, dofs, kinematics, material)
disp = sample_field(mech.displacement, points[node_mask])

disp_full = np.zeros((points.shape[0], 3))
disp_full[node_mask] = disp
displacement = disp_full.reshape(*nnodes, 3)
print(f"\tsampled {int(node_mask.sum())}/{node_mask.size} solid nodes", flush=True)

# nodal displacement only; indicator is geometry (data/abc/geometry/voxel/<name>.npz)
# and stress is derivable from the nodal field, so neither is stored here.
disp_dir = SOLUTION_DIR / "displacement"
disp_dir.mkdir(parents=True, exist_ok=True)
# spacing is omitted (derivable as Lx/Ncells, Ncells = nodal axis length - 1)
meta = dict(origin=origin_v, Lx=lengths_v[0], Ly=lengths_v[1], Lz=lengths_v[2])
disp_out = disp_dir / f"{name}_{args.case}.npz"
np.savez_compressed(
    disp_out, displacement=to_float16(displacement, "displacement"), **meta
)
print(f"\t-> {disp_out}", flush=True)

if POSTPROCESSING:
    RESULTS_SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_SOLUTION_DIR / f"{name}_{args.case}_voxel"
    imageToVTK(
        str(out),
        origin=tuple(
            float(x) for x in origin_v
        ),  # plain floats: np.float64 str() breaks the .vti
        spacing=(float(spacing),) * D,
        cellData={"indicator": np.ascontiguousarray(indicator)},
        pointData={
            "displacement": tuple(
                np.ascontiguousarray(displacement[..., i]) for i in range(D)
            )
        },
    )
    print(f"\t-> {out}.vti", flush=True)

# ----------------------------------- nodal fixture -----------------------------------
# A boundary grip constrains/loads the solid FACES it covers. Tag a face when its owning
# boundary cell lies under the grip (rasterize) AND its outward normal aligns with the
# grip's local normal -- the same orientation test elastic_create_fixture.py uses to pick
# the grip triangles. Stamping per exposed face (not promoting each cut cell to all its
# corner nodes) tags exactly the surface node-plane on the grip and never the boundary
# cell's perpendicular side faces, which a cell->node promotion would wrongly load.
COS_TOL = 0.3
# A grip on the major (pull) axis sits on the bounding-box end, which on a zero-padding
# axis (Lv == extent) coincides with the voxel mesh's outer face -- intersectWithMesh
# returns nothing for a surface flush with the boundary. Inflate the rasterization grid
# by a hair (same 1e-9 trick the FCM grid uses above) so such grips land just inside.
EPS_MARGIN = 1e-9
voxgrid = mlhp.makeRefinedGrid(
    list(ncells),
    [float(x) + 2 * EPS_MARGIN for x in lengths_v],
    [float(x) - EPS_MARGIN for x in origin_v],
)
kept = np.argwhere(solid)
dirichlet_mask = np.zeros((*nnodes, D), dtype=np.uint8)
dirichlet_value = np.zeros((*nnodes, D))
neumann = np.zeros((*nnodes, D))
for b in spec["boundaries"]:
    grip = mlhp.readStl(str(FIXTURE_DIR / b["stl"]))
    cut = rasterize(grip, voxgrid, ncells)  # boundary cells under the grip
    gverts = np.asarray(grip.vertices, dtype=np.float64)
    gcells = np.asarray(grip.cells, dtype=np.int64)
    gnormals = np.asarray(grip.normals, dtype=np.float64)
    gtree = cKDTree(gverts[gcells].mean(axis=1))
    tag = np.zeros(nnodes, dtype=bool)
    for axis in range(D):
        for side in (0, 1):
            normal = np.zeros(D)
            normal[axis] = 1.0 if side == 1 else -1.0
            face_nodes = [l for l in range(8) if OFFSET[l, axis] == side]
            nb = kept.copy()
            nb[:, axis] += 1 if side == 1 else -1
            inb = (nb[:, axis] >= 0) & (nb[:, axis] < ncells[axis])
            nbsolid = np.zeros(len(kept), dtype=bool)
            nbsolid[inb] = solid[nb[inb, 0], nb[inb, 1], nb[inb, 2]]
            face = kept[~nbsolid & cut[kept[:, 0], kept[:, 1], kept[:, 2]]]
            if not len(face):
                continue
            fc = origin_v + spacing * (
                face + 0.5 + 0.5 * normal
            )  # (axis, side) face center
            _, gi = gtree.query(fc)
            face = face[
                (gnormals[gi] @ normal) > COS_TOL
            ]  # keep grip-aligned faces only
            if not len(face):
                continue
            fn = (face[:, None, :] + OFFSET[None, face_nodes, :]).reshape(-1, 3)
            tag[fn[:, 0], fn[:, 1], fn[:, 2]] = True
    if b["type"] == "dirichlet":
        dirichlet_mask[tag] = 1
        dirichlet_value[tag] = b["value"]
    else:
        neumann[tag] = b["value"]
print(
    f"\tfixture {int((dirichlet_mask[..., 0] > 0).sum())} dirichlet,"
    f" {int(np.any(neumann != 0, axis=-1).sum())} neumann surface nodes",
    flush=True,
)

voxel_dir = FIXTURE_DIR / "boundary_voxel"
voxel_dir.mkdir(parents=True, exist_ok=True)
out = voxel_dir / f"{name}_{args.case}.npz"
# indicator is omitted (it's geometry, in data/abc/geometry/voxel/<name>.npz);
# this file keeps only the BC fields plus the grid metadata to locate them.
np.savez_compressed(
    out,
    dirichlet_mask=dirichlet_mask,
    dirichlet_value=dirichlet_value.astype(np.float32),
    neumann=neumann.astype(np.float32),
    origin=origin_v,
    Lx=lengths_v[0],
    Ly=lengths_v[1],
    Lz=lengths_v[2],
)
print(f"\t-> {out}", flush=True)

if POSTPROCESSING:
    RESULTS_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_FIXTURE_DIR / f"{name}_{args.case}"
    imageToVTK(
        str(out),
        origin=tuple(
            float(x) for x in origin_v
        ),  # plain floats: np.float64 str() breaks the .vti
        spacing=(float(spacing),) * D,
        cellData={"indicator": np.ascontiguousarray(indicator)},
        pointData={
            "dirichlet": tuple(
                np.ascontiguousarray(dirichlet_mask[..., i]) for i in range(D)
            ),
            "neumann": tuple(np.ascontiguousarray(neumann[..., i]) for i in range(D)),
        },
    )
    print(f"\t-> {out}.vti", flush=True)
