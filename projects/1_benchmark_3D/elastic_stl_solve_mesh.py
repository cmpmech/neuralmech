import argparse
import json
from pathlib import Path

import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
MESH_DIR = DATA_DIR / "geometry/mesh"
FIXTURE_DIR = DATA_DIR / "elasticity/fixture"
RESULTS_SOLUTION_DIR = (
    BASE_DIR / "../../results/abc/elasticity/solution_stl"
).resolve()

D = 3

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--case", type=int, default=1)
args = parser.parse_args()

# ----------------------------------- solver settings ---------------------------------
RTOL = 1e-8
MAXITER = 5000
COS_TOL = 0.3  # min |cos| between a boundary-face normal and the grip's outward normal

POSTPROCESSING = True

name = f"{args.geometry:09}_abc"


# ----------------------------------- helpers -----------------------------------------
def boundary_faces(tets, vertices):  # outward-oriented surface triangles of a tet mesh
    faces = tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    opposite = tets[:, [3, 2, 1, 0]].reshape(-1)  # the 4th vertex of each face's tet
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    on_boundary = counts[inverse.ravel()] == 1  # a face on the hull belongs to one tet
    faces, opposite = faces[on_boundary], opposite[on_boundary]
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    normal = np.cross(v1 - v0, v2 - v0)
    center = (v0 + v1 + v2) / 3.0
    flip = np.einsum("ij,ij->i", normal, center - vertices[opposite]) < 0  # face outward
    normal[flip] *= -1
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    return faces, center, normal / length, 0.5 * length[:, 0]  # faces, centroid, n, area


# ----------------------------------- problem setup -----------------------------------
spec = json.loads((FIXTURE_DIR / f"{name}_{args.case}.json").read_text())
data = np.load(MESH_DIR / f"{name}.npz")
vertices, tets = data["vertices"], data["cells"].reshape(-1, 4)
mesh = mlhp.makeUnstructuredMesh(
    data["vertices"], data["cells"], data["offsets"], filter=False, reorder=False
)
basis = mlhp.makeUnstructuredBasis(mesh, nfields=D)

# node -> global dofs: locationMaps are field-major per element ([f, local node]); each
# mesh node owns D consecutive dofs. reorder=False keeps node index == vertex index.
lmaps = np.array(basis.locationMaps()).reshape(len(tets), D, 4)
node_dof = np.empty((len(vertices), D), dtype=np.int64)
node_dof[tets] = np.transpose(lmaps, (0, 2, 1))

E = spec["material"]["E"]
nu = spec["material"]["nu"]
kinematics = mlhp.smallStrainKinematics(D)
material = mlhp.isotropicElasticMaterial(
    mlhp.scalarField(D, E), mlhp.scalarField(D, nu)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, material, mlhp.vectorField(D, [0.0] * D)
)

# ----------------------------------- boundary conditions -----------------------------
# Each grip is a thin STL slab covering a surface patch. Tag the mesh boundary faces
# inside the grip's (inflated) box whose outward normal aligns with the grip's normal --
# the same orientation test elastic_create_fixture.py / elastic_stl_solve.py use. Tagged
# Dirichlet faces become strong nodal constraints; Neumann faces become a consistent
# nodal traction (traction * area / 3 to each face node).
faces, fcenter, fnormal, farea = boundary_faces(tets, vertices)
margin = np.sqrt(2.0 * farea).mean()  # ~ one boundary-face edge length

didx, dval = [], []
fext = np.zeros((len(vertices), D))
for b in spec["boundaries"]:
    grip = mlhp.readStl(str(FIXTURE_DIR / b["stl"]))
    gverts = np.asarray(grip.vertices, dtype=np.float64)
    gnormal = np.asarray(grip.normals, dtype=np.float64).mean(axis=0)
    gnormal /= np.linalg.norm(gnormal)
    inside_box = np.all(
        (fcenter >= gverts.min(0) - margin) & (fcenter <= gverts.max(0) + margin),
        axis=1,
    )
    tagged = inside_box & (fnormal @ gnormal > COS_TOL)
    if b["type"] == "dirichlet":
        nodes = np.unique(faces[tagged])
        for f in range(D):
            didx.extend(node_dof[nodes, f].tolist())
            dval.extend([b["value"][f]] * len(nodes))
    else:
        load = np.asarray(b["value"], dtype=np.float64)
        np.add.at(fext, faces[tagged], (farea[tagged, None] * load / 3.0)[:, None, :])

didx = np.asarray(didx, dtype=np.int64)
unique_dofs, first = np.unique(didx, return_index=True)
dirichlet = [unique_dofs.tolist(), np.asarray(dval)[first].tolist()]
print(
    f"\tBCs: {len(unique_dofs)} dirichlet dofs, "
    f"{int(np.any(fext != 0, axis=1).sum())} loaded nodes",
    flush=True,
)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

# add the Neumann nodal forces to the (condensed) interior right-hand side
ndof = basis.ndof()
interior_mask = np.ones(ndof, dtype=bool)
interior_mask[unique_dofs] = False
global_to_interior = np.full(ndof, -1, dtype=np.int64)
global_to_interior[interior_mask] = np.arange(interior_mask.sum())
rhs = np.array(vector)
gdof = node_dof.ravel()
keep = interior_mask[gdof]
np.add.at(rhs, global_to_interior[gdof[keep]], fext.ravel()[keep])
vector = mlhp.DoubleVector(rhs.tolist())

# -------------------------------------- solve ----------------------------------------
P = mlhp.diagonalPreconditioner(matrix)
interior, norms = mlhp.cg(
    matrix, vector, M=P, rtol=RTOL, maxiter=MAXITER, residualNorms=True
)
dofs = mlhp.inflateDofs(interior, dirichlet)
print(
    f"{name} case {args.case} ({spec['label']}): "
    f"{len(norms)} CG iters, residual {norms[-1]:.2e}, "
    f"max |u| {np.max(np.abs(dofs)):.3e}",
    flush=True,
)

# ----------------------------------- postprocessing ----------------------------------
if POSTPROCESSING:
    processors = [
        mlhp.solutionProcessor(D, dofs, "Displacement"),
        mlhp.stressProcessor(dofs, kinematics, material),
        mlhp.vonMisesProcessor(dofs, kinematics, material, "VonMises"),
    ]
    postmesh = mlhp.gridCellMesh([1] * D)
    RESULTS_SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_SOLUTION_DIR / f"{name}_{args.case}_mesh"
    mlhp.basisOutput(basis, postmesh, mlhp.PVtuOutput(filename=str(out)), processors)
    print(f"\t-> {out}.pvtu", flush=True)
