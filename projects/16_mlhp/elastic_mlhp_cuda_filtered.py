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

DEGREE = 1
ALPHA = 1e-4  # does not matter (filtered out)
FILTER_THRESHOLD = 0  # uint8; elements with indicator <= threshold are removed

DTYPE = cp.float32  # cp.float32 for single precision
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
CG_TOL = 1e-6 if DTYPE == cp.float32 else 1e-10

BLOCK = 1024
PRECOMPILED = False
compiler_options = ()  # flags don't seem to help
# compiler_options = ("--use_fast_math", "--gpu-architecture=compute_120")

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
    nelems = [Nx, Ny]
    domain_lengths = [Lx, Ly]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    nelems = [Nx, Ny, Nz]
    domain_lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

keep_mask = indicator.ravel("C") > FILTER_THRESHOLD
N_elems = int(keep_mask.sum())
indicator_filtered = indicator.ravel("C")[keep_mask]  # only kept elements, C-order
print(
    f"elements: {indicator.size} total, {N_elems} kept ({100 * N_elems / indicator.size:.1f}%)"
)

nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(
        mlhp.makeGrid(ncells=nelems, lengths=domain_lengths), mask=keep_mask
    )
)
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

# ------------------------ local preintegrated stiffness matrix -----------------------
tic = time.time()
kinematics = mlhp.smallStrainKinematics(D)
constitutive = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, constitutive, mlhp.vectorField(D, [0.0] * D)
)

mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis_local = mlhp.makeHpTrunkSpace(mesh_local, degree=DEGREE, nfields=D)
matrix_local = mlhp.allocateSparseMatrix(basis_local)
rhs_local = mlhp.allocateRhsVector(matrix_local)
mlhp.integrateOnDomain(
    basis_local,
    integrand,
    [matrix_local, rhs_local],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
)
K_local = np.array(matrix_local.todense())
print(f"preintegration: {time.time() - tic:.2f}s")

# -------------------------------------- assembly -------------------------------------
tic = time.time()
# allocateSparseMatrix is only needed to size the condensed vector; it is never filled
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
print(f"assembly: {time.time() - tic:.2f}s")

# --------------------------------------- cuda ----------------------------------------
cuda_source = (BASE_DIR / "../../solvers/kernels/mlhp_kernels.cu").read_text()
cuda_options = (("-DUSE_FLOAT",) if DTYPE == cp.float32 else ()) + compiler_options

if PRECOMPILED:
    ptx_stem = "mlhp_kernels_f32" if DTYPE == cp.float32 else "mlhp_kernels_f64"
    module = cp.RawModule(
        path=str(BASE_DIR / f"../../solvers/kernels_build/{ptx_stem}.cubin")
    )
else:
    module = cp.RawModule(code=cuda_source, options=cuda_options)
Ku_kernel = module.get_function("Ku_kernel")
K_diag_kernel = module.get_function("K_diag_kernel")

# for indexing in kernel
ndof_e = K_local.shape[0]
grid = (N_elems + BLOCK - 1) // BLOCK

E_scalar = np_dtype(E)
alpha_scalar = np_dtype(ALPHA)

cp.cuda.Stream.null.synchronize()
tic = time.time()
K_local_gpu = cp.array(K_local.ravel("C"), dtype=DTYPE)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
indicator_gpu = cp.array(indicator_filtered, dtype=cp.uint8)
rhs_gpu = cp.array(rhs, dtype=DTYPE)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)
cp.cuda.Stream.null.synchronize()
print(f"GPU upload: {time.time() - tic:.3f}s")

# ------------------------------------ cuda kernels -----------------------------------
K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
cp.cuda.Stream.null.synchronize()
tic = time.time()
K_diag_kernel(
    (grid,),
    (BLOCK,),
    (
        K_diag_gpu,
        indicator_gpu,
        efts_gpu,
        K_local_gpu,
        E_scalar,
        alpha_scalar,
        N_elems,
        ndof_e,
    ),
)
K_diag_gpu[constrained_gpu] = 1.0
cp.cuda.Stream.null.synchronize()
print(f"K_diag: {time.time() - tic:.3f}s")


def get_Ku(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=DTYPE)
    Ku_kernel(
        (grid,),
        (BLOCK,),
        (
            u_gpu,
            Ku_gpu,
            indicator_gpu,
            efts_gpu,
            K_local_gpu,
            E_scalar,
            alpha_scalar,
            N_elems,
            ndof_e,
        ),
    )
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
    nvoxels=nelems,
    lengths=domain_lengths,
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)

out = str(RESULTS_DIR / f"elastic_mlhp_cuda_filtered_{ct_file[:-4]}")
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
