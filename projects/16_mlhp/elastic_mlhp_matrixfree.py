import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse.linalg

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = args.dim

DEGREE = 1
ALPHA = 1e-5

E = 210.0
NU = 0.3
FORCE = 1.0

# ------------------------------------ ct geometry ------------------------------------
ct_file = args.ct if args.ct else f"plate_hole_{D}D.npz"
ct = np.load(DATA_DIR / ct_file)
indicator = ct["indicator"]  # uint8

if D == 2:
    Lx, Ly = float(ct["Lx"]), float(ct["Ly"])
    Nx, Ny = indicator.shape
    ncells = [Nx, Ny]
    lengths = [Lx, Ly]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

E_values = E * np.maximum(indicator.ravel("C") / 255.0, ALPHA)
nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells, lengths=lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
ndof = basis.ndof()
print(basis)

# -------------------------------- boundary conditions --------------------------------
# uni-axial tension
bc_faces = [0, 2] if D == 2 else [0, 2, 4]  # with x, y, (z) constrained
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2
    )
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained_dofs = np.array(dirichlet[0])

# ------------------------ local preintegrated stiffness matrix -----------------------
kinematics = mlhp.smallStrainKinematics(D)
constitutive = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, constitutive, mlhp.vectorField(D, [0.0] * D)
)

tic = time.time()
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis_local = mlhp.makeHpTrunkSpace(mesh_local, degree=DEGREE, nfields=D)

matrix_local = mlhp.allocateSparseMatrix(basis_local)  # no bcs
rhs_local = mlhp.allocateRhsVector(
    matrix_local
)  # NOT NEEDED CURRENTLY, AS NO BODY LOAD -> generalize

mlhp.integrateOnDomain(
    basis_local,
    integrand,
    [matrix_local, rhs_local],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
)

K_local = np.array(matrix_local.todense())
K_local_diag = np.diag(K_local)
print(f"preintegration: {time.time() - tic:.2f}s")
# -------------------------------------- assembly -------------------------------------
tic = time.time()
# allocateSparseMatrix is only needed to size the condensed vector: it is never filled
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
del matrix

traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)

interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = vector.array
del vector

efts = np.array(basis.locationMaps())
efts_flat = efts.ravel()

print(f"assembly: {time.time() - tic:.2f}s")


# --------------------------------------- solve ---------------------------------------
def get_Ku(u):
    Ku_local = E_values[:, None] * (u[efts] @ K_local)
    Ku = np.bincount(efts_flat, weights=Ku_local.ravel(), minlength=ndof)
    Ku[constrained_dofs] = u[constrained_dofs]
    return Ku


K_diag = np.bincount(
    efts_flat,
    weights=(E_values[:, None] * K_local_diag[None, :]).ravel(),
    minlength=ndof,
)
K_diag[constrained_dofs] = 1.0

KU_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=get_Ku)
K_diag_op = scipy.sparse.linalg.LinearOperator(
    (ndof, ndof), matvec=lambda v: v / K_diag
)

iters = [0]


def callback(x):
    iters[0] += 1


tic = time.time()
sol, info = scipy.sparse.linalg.cg(
    KU_op, rhs, M=K_diag_op, rtol=1e-10, maxiter=20000, callback=callback
)
print(f"CG: {iters[0]} iterations, info={info}, {time.time() - tic:.2f}s")
print(f"max displacement: {np.max(np.abs(sol)):.3e}")

# --------------------------------------- export --------------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
    nvoxels=ncells,
    lengths=lengths,
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)

out = str(RESULTS_DIR / f"elastic_mlhp_{ct_file[:-4]}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out}.pvtu")

# ----------------------------------- postprocessing ----------------------------------
if D == 2:
    result = mlhp.DataAccumulator()
    mlhp.basisOutput(basis, postmesh, result, processors)
    ux = np.array(result.data()[0])[0::2]

    tri = result.triangulation()
    ind_viz = np.array(result.data()[1])
    tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)

    fig, ax = plt.subplots()
    cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
    fig.colorbar(cb)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()
