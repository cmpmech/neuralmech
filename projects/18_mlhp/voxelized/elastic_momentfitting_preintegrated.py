import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg
import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../../results/3D").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

DEGREE = 3
ALPHA = 1e-4
SUB_VOXELS = 32  # 128  # 32  # 40

DTYPE = cp.float64
CG_TOL = 1e-10
BLOCK = 1024

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
    nvoxels = [Nx, Ny]
    domain_lengths = [Lx, Ly]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    nvoxels = [Nx, Ny, Nz]
    domain_lengths = [Lx, Ly, Lz]

assert all(n % SUB_VOXELS == 0 for n in nvoxels), (
    f"grid dims {nvoxels} must all be divisible by SUB_VOXELS={SUB_VOXELS}"
)

nelems = [n // SUB_VOXELS for n in nvoxels]
N_elems = int(np.prod(nelems))

E_float = np.maximum(indicator.astype(np.float64) / 255.0, ALPHA)
nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems, lengths=domain_lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
ndof = basis.ndof()
print(basis)

# -------------------------------- boundary conditions --------------------------------
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2
    )
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained_dofs = np.array(dirichlet[0])

# ----------------------------------- preintegration ----------------------------------
kinematics = mlhp.smallStrainKinematics(D)
material = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)

elem_lengths = [domain_lengths[d] / nelems[d] for d in range(D)]
detJ = float(np.prod([length / 2.0 for length in elem_lengths]))

mesh_ref = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis_ref = mlhp.makeHpTrunkSpace(mesh_ref, degree=DEGREE, nfields=D)
integrand = mlhp.staticDomainIntegrand(
    kinematics, material, mlhp.vectorField(D, [0.0] * D)
)

tic = time.time()
M = np.array(mlhp.voxelMomentFittingMatrix([SUB_VOXELS] * D, DEGREE))
Bmat = np.array(mlhp.momentFittingPointMatrices(basis_ref, integrand, DEGREE))
ndof_e = basis_ref.ndof()
print(f"preintegration: {time.time() - tic:.2f}s")

n_mf = (2 * DEGREE + 1) ** D
assert M.shape == (n_mf, SUB_VOXELS**D), M.shape
assert Bmat.shape == (n_mf, ndof_e, ndof_e), Bmat.shape

if D == 2:
    E_elem = (
        (E * E_float)
        .reshape(nelems[0], SUB_VOXELS, nelems[1], SUB_VOXELS)
        .transpose(0, 2, 1, 3)
        .reshape(N_elems, SUB_VOXELS**D)
    )
else:
    E_elem = (
        (E * E_float)
        .reshape(nelems[0], SUB_VOXELS, nelems[1], SUB_VOXELS, nelems[2], SUB_VOXELS)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(N_elems, SUB_VOXELS**D)
    )

# -------------------------------------- assembly -------------------------------------
tic = time.time()
weights_elem = detJ * (
    E_elem @ M.T
)
K_e = np.einsum("ep,pij->eij", weights_elem, Bmat)
efts = np.array(basis.locationMaps())

# load vector: Neumann traction on the right face, expanded to the full dof space
matrix_tmp = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix_tmp)
del matrix_tmp
traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)

interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = vector.array
del vector
print(f"assembly: {time.time() - tic:.2f}s")

# ---------------------------------------- cuda ---------------------------------------
cuda_source = (BASE_DIR / "../../../solvers/kernels/mlhp_kernels.cu").read_text()
cuda_options = ("-DUSE_FLOAT",) if DTYPE == cp.float32 else ()
module = cp.RawModule(code=cuda_source, options=cuda_options)
Ku_kernel = module.get_function("Ku_subvoxel_kernel")
K_diag_kernel = module.get_function("K_diag_subvoxel_kernel")

grid = (N_elems + BLOCK - 1) // BLOCK

cp.cuda.Stream.null.synchronize()
tic = time.time()
K_e_gpu = cp.array(K_e.ravel("C"), dtype=DTYPE)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
rhs_gpu = cp.array(rhs, dtype=DTYPE)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)
cp.cuda.Stream.null.synchronize()
print(f"GPU upload: {time.time() - tic:.3f}s")

# ------------------------------------ cuda kernels -----------------------------------
K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
K_diag_kernel((grid,), (BLOCK,), (K_diag_gpu, efts_gpu, K_e_gpu, N_elems, ndof_e))
K_diag_gpu[constrained_gpu] = 1.0


def get_Ku(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=DTYPE)
    Ku_kernel((grid,), (BLOCK,), (u_gpu, Ku_gpu, efts_gpu, K_e_gpu, N_elems, ndof_e))
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu


KU_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=get_Ku)
K_diag_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

# --------------------------------------- solve ---------------------------------------
iters = [0]


def callback(x):
    iters[0] += 1


cp.cuda.Stream.null.synchronize()
tic = time.time()
sol_gpu, info = cp_splinalg.cg(
    KU_op, rhs_gpu, M=K_diag_op, rtol=CG_TOL, maxiter=20000, callback=callback
)
cp.cuda.Stream.null.synchronize()
print(f"CG: {iters[0]} iterations, info={info}, {time.time() - tic:.2f}s")

sol = sol_gpu.get()
print(f"max displacement: {np.max(np.abs(sol)):.3e}")

# --------------------------------------- export --------------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
    nvoxels=nvoxels,
    lengths=domain_lengths,
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)

out = str(RESULTS_DIR / f"elastic_mlhp_momentfitting_preintegrated_{ct_file[:-4]}")
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
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
