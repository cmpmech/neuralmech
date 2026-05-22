import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg
import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = args.dim

DEGREE = 3
ALPHA = 1e-5
SUB_VOXELS = 8

DTYPE = cp.float64  # cp.float32 for single precision
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
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
    ncells = [Nx, Ny]
    lengths = [Lx, Ly]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

assert all(n % SUB_VOXELS == 0 for n in ncells), (
    f"all grid dimensions must be divisible by SUB_VOXELS={SUB_VOXELS}"
)

ncells_elem = [n // SUB_VOXELS for n in ncells]
macro_lengths = [SUB_VOXELS * l for l in elem_lengths]

nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells_elem, lengths=lengths))
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

# ----------------------- subvoxel preintegrated reference matrices -------------------
kinematics = mlhp.smallStrainKinematics(D)
mesh_ref = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=macro_lengths))
basis_ref = mlhp.makeHpTrunkSpace(mesh_ref, degree=DEGREE, nfields=D)
ndof_e = basis_ref.ndof()
de = mlhp.combineDirichletDofs([])

n_sub = SUB_VOXELS**D
K_refs = np.zeros((n_sub, ndof_e, ndof_e))

tic = time.time()
for s in range(n_sub):
    ind_s = np.zeros(n_sub, dtype=np.float32)
    ind_s[s] = 1.0
    E_field_s = mlhp.scalarFieldFromVoxelData(
        mlhp.FloatVector(ind_s.tolist()),
        nvoxels=[SUB_VOXELS] * D,
        lengths=macro_lengths,
    )
    c_s = (
        mlhp.planeStressMaterial(E_field_s, nu_field)
        if D == 2
        else mlhp.isotropicElasticMaterial(E_field_s, nu_field)
    )
    i_s = mlhp.staticDomainIntegrand(kinematics, c_s, mlhp.vectorField(D, [0.0] * D))
    m_s = mlhp.allocateSparseMatrix(basis_ref, de[0])
    v_s = mlhp.allocateRhsVector(m_s)
    mlhp.integrateOnDomain(
        basis_ref,
        i_s,
        [m_s, v_s],
        quadrature=mlhp.gridQuadrature(nsubcells=[SUB_VOXELS] * D),
        dirichletDofs=de,
    )
    K_refs[s] = np.array(m_s.todense())
print(f"K_refs ({n_sub}): {time.time() - tic:.2f}s")

# -------------------------------------- assembly -------------------------------------
c_rhs = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
i_rhs = mlhp.staticDomainIntegrand(kinematics, c_rhs, mlhp.vectorField(D, [0.0] * D))
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

tic = time.time()
mlhp.integrateOnDomain(
    basis,
    i_rhs,
    [matrix, vector],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
    dirichletDofs=dirichlet,
)
traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
print(f"assembly: {time.time() - tic:.2f}s")

interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = np.array(list(vector))
del matrix, vector

efts = np.array(basis.locationMaps())

# --------------------------------------- cuda ----------------------------------------
cuda_source = (BASE_DIR / "mlhp_kernels.cu").read_text()
cuda_options = ("-DUSE_FLOAT",) if DTYPE == cp.float32 else ()
module = cp.RawModule(code=cuda_source, options=cuda_options)
kernel_matvec = module.get_function("cuda_matvec_sub")
kernel_k_diag = module.get_function("cuda_k_diag_sub")

n_elem = int(np.prod(ncells_elem))
grid = (n_elem + BLOCK - 1) // BLOCK

E_scalar = np_dtype(E)
alpha_scalar = np_dtype(ALPHA)
Ny_elem = ncells_elem[1]
Nz_elem = ncells_elem[2] if D == 3 else 1
Ny_vox = ncells[1]
Nz_vox = ncells[2] if D == 3 else 1
sub_voxels_z = SUB_VOXELS if D == 3 else 1

cp.cuda.Stream.null.synchronize()
tic = time.time()
K_refs_gpu = cp.array(K_refs.ravel("C"), dtype=DTYPE)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
indicator_gpu = cp.array(indicator.ravel("C"), dtype=cp.uint8)
rhs_gpu = cp.array(rhs, dtype=DTYPE)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)
cp.cuda.Stream.null.synchronize()
print(f"GPU upload: {time.time() - tic:.3f}s")

sub_args = (n_sub, SUB_VOXELS, sub_voxels_z, Ny_elem, Nz_elem, Ny_vox, Nz_vox)

K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
cp.cuda.Stream.null.synchronize()
tic = time.time()
kernel_k_diag(
    (grid,),
    (BLOCK,),
    (
        K_diag_gpu,
        indicator_gpu,
        efts_gpu,
        K_refs_gpu,
        E_scalar,
        alpha_scalar,
        n_elem,
        ndof_e,
        *sub_args,
    ),
)
K_diag_gpu[constrained_gpu] = 1.0
cp.cuda.Stream.null.synchronize()
print(f"K_diag: {time.time() - tic:.3f}s")


def matvec_gpu(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=DTYPE)
    kernel_matvec(
        (grid,),
        (BLOCK,),
        (
            u_gpu,
            Ku_gpu,
            indicator_gpu,
            efts_gpu,
            K_refs_gpu,
            E_scalar,
            alpha_scalar,
            n_elem,
            ndof_e,
            *sub_args,
        ),
    )
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu


A_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=matvec_gpu)
P_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

# --------------------------------------- solve ---------------------------------------
iters = [0]


def callback(x):
    iters[0] += 1


cp.cuda.Stream.null.synchronize()
tic = time.time()
sol_gpu, info = cp_splinalg.cg(
    A_op, rhs_gpu, M=P_op, tol=1e-10, maxiter=20000, callback=callback
)
cp.cuda.Stream.null.synchronize()
print(f"CG: {iters[0]} iterations, info={info}, {time.time() - tic:.2f}s")

sol = sol_gpu.get()
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

out = str(RESULTS_DIR / f"elastic_mlhp_cuda_sub{SUB_VOXELS}_{ct_file[:-4]}")
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
