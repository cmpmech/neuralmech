import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.ndimage as cnd
import cupyx.scipy.sparse.linalg as cp_splinalg
import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

DEGREE = 3
SUB_VOXELS = 5  # design (voxel) resolution per coarse finite element
QUAD_ORDER = DEGREE + 1

VOLFRAC = 0.5
PENAL = 3.0
# RMIN = 1.5 * SUB_VOXELS  # density-filter radius in voxel units
RMIN = 1.5  # voxels
MOVE = 0.2
MAX_ITER = 120
CHANGE_TOL = 0.01

E0 = 1.0
EMIN = 1e-6
NU = 0.3
TRACTION = 1.0

DTYPE = cp.float64  # cp.float32 for single precision
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
CG_TOL = 1e-6 if DTYPE == cp.float32 else 1e-10

BLOCK = 1024

if D == 2:
    NVOXELS = [210, 70]
    # NVOXELS = [90, 30]
    DOMAIN_LENGTHS = [3.0, 1.0]
else:
    NVOXELS = [96, 32, 32]
    DOMAIN_LENGTHS = [3.0, 1.0, 1.0]

assert all(n % SUB_VOXELS == 0 for n in NVOXELS)
nelems = [n // SUB_VOXELS for n in NVOXELS]
elem_lengths = [l / n for l, n in zip(DOMAIN_LENGTHS, nelems)]
n_voxels = int(np.prod(NVOXELS))

nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems, lengths=DOMAIN_LENGTHS))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
ndof = basis.ndof()
print(basis)

# -------------------------------- boundary conditions --------------------------------
zero = mlhp.scalarField(D, 0.0)
bc_symmetry = mlhp.integrateDirichletDofs(zero, basis, [0], ifield=0)

bottom = set(mlhp.integrateDirichletDofs(zero, basis, [2], ifield=1)[0])
right = set(mlhp.integrateDirichletDofs(zero, basis, [1], ifield=1)[0])
roller_dofs = sorted(bottom & right)
bc_roller = (roller_dofs, [0.0] * len(roller_dofs))

bc_list = [bc_symmetry, bc_roller]
if D == 3:
    bc_list.append(mlhp.integrateDirichletDofs(zero, basis, [4], ifield=2))
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained_dofs = np.array(dirichlet[0])

# ----------------------- local preintegrated stiffness matrices ----------------------
kinematics = mlhp.smallStrainKinematics(D)
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis_local = mlhp.makeHpTrunkSpace(mesh_local, degree=DEGREE, nfields=D)
ndof_e = basis_local.ndof()
n_sub = SUB_VOXELS**D

material = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, material, mlhp.vectorField(D, [0.0] * D)
)
quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS] * D)

tic = time.time()
K_refs = mlhp.integratePartitionMatrices(
    basis_local, integrand, quadrature, mlhp.absoluteQuadratureOrder([QUAD_ORDER] * D)
)
print(f"preintegration ({n_sub} subvoxels): {time.time() - tic:.2f}s")

# ------------------------------------ load vector ------------------------------------
load_width = elem_lengths[0]
if D == 2:
    load_expr = f"[0.0, -{TRACTION} if x < {load_width} else 0.0]"
else:
    load_expr = f"[0.0, -{TRACTION} if x < {load_width} else 0.0, 0.0]"
neumann = mlhp.neumannIntegrand(mlhp.vectorField(D, load_expr))
load_quad = mlhp.quadratureOnMeshFaces(mesh, [3])

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
del matrix
mlhp.integrateOnSurface(basis, neumann, [vector], load_quad, dirichletDofs=dirichlet)

interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = vector.array
del vector

efts = np.array(basis.locationMaps())

# ---------------------------------------- cuda ---------------------------------------
cuda_source = (BASE_DIR / "../../solvers/kernels/mlhp_kernels.cu").read_text()
cuda_options = ("-DUSE_FLOAT",) if DTYPE == cp.float32 else ()
module = cp.RawModule(code=cuda_source, options=cuda_options)
assemble_K_e_kernel = module.get_function("assemble_K_e_density_kernel")
sensitivity_kernel = module.get_function("compliance_sensitivity_kernel")
Ku_kernel = module.get_function("Ku_subvoxel_kernel")
K_diag_kernel = module.get_function("K_diag_subvoxel_kernel")

N_elems = int(np.prod(nelems))
Ny_elem, Nz_elem = nelems[1], nelems[2] if D == 3 else 1
Ny_vox, Nz_vox = NVOXELS[1], NVOXELS[2] if D == 3 else 1
Sz = SUB_VOXELS if D == 3 else 1
grid = (N_elems + BLOCK - 1) // BLOCK

K_refs_gpu = cp.array(K_refs.ravel("C"), dtype=DTYPE)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
rhs_gpu = cp.array(rhs, dtype=DTYPE)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)

K_e_gpu = cp.empty(N_elems * ndof_e * ndof_e, dtype=DTYPE)

# --------------------------------------- filter --------------------------------------
ceil_r = int(np.ceil(RMIN))
axes = np.meshgrid(*([np.arange(-ceil_r, ceil_r + 1)] * D), indexing="ij")
h = np.maximum(0.0, RMIN - np.sqrt(sum(a**2 for a in axes)))
h_gpu = cp.array(h, dtype=DTYPE)
Hs_gpu = cnd.convolve(cp.ones(NVOXELS, dtype=DTYPE), h_gpu, mode="constant")

density_filter = lambda x: cnd.convolve(x, h_gpu, mode="constant") / Hs_gpu
sens_filter = lambda g: cnd.convolve(g / Hs_gpu, h_gpu, mode="constant")


def solve(E_voxels, u0):
    assemble_K_e_kernel(
        (grid,),
        (BLOCK,),
        (
            K_e_gpu,
            E_voxels.ravel(),
            K_refs_gpu,
            N_elems,
            ndof_e,
            n_sub,
            SUB_VOXELS,
            Sz,
            Ny_elem,
            Nz_elem,
            Ny_vox,
            Nz_vox,
        ),
    )
    K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
    K_diag_kernel((grid,), (BLOCK,), (K_diag_gpu, efts_gpu, K_e_gpu, N_elems, ndof_e))
    K_diag_gpu[constrained_gpu] = 1.0

    def get_Ku(u):
        Ku = cp.zeros(ndof, dtype=DTYPE)
        Ku_kernel((grid,), (BLOCK,), (u, Ku, efts_gpu, K_e_gpu, N_elems, ndof_e))
        Ku[constrained_gpu] = u[constrained_gpu]
        return Ku

    KU_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=get_Ku)
    M_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)
    iters = [0]
    u, _ = cp_splinalg.cg(
        KU_op,
        rhs_gpu,
        x0=u0,
        M=M_op,
        rtol=CG_TOL,
        maxiter=20000,
        callback=lambda x: iters.__setitem__(0, iters[0] + 1),
    )
    return u, iters[0]


def sensitivity(u):
    ce = cp.zeros(n_voxels, dtype=DTYPE)
    sensitivity_kernel(
        (grid,),
        (BLOCK,),
        (
            u,
            ce,
            efts_gpu,
            K_refs_gpu,
            N_elems,
            ndof_e,
            n_sub,
            SUB_VOXELS,
            Sz,
            Ny_elem,
            Nz_elem,
            Ny_vox,
            Nz_vox,
        ),
    )
    return ce.reshape(NVOXELS)


# ------------------------------------ optimization -----------------------------------
rho = cp.full(NVOXELS, VOLFRAC, dtype=DTYPE)
u_gpu = cp.zeros(ndof, dtype=DTYPE)
dv = cp.ones(NVOXELS, dtype=DTYPE)

tic = time.time()
for loop in range(1, MAX_ITER + 1):
    rho_phys = density_filter(rho)
    E_voxels = EMIN + rho_phys**PENAL * (E0 - EMIN)

    u_gpu, cg_iters = solve(E_voxels, u_gpu)
    ce = sensitivity(u_gpu)

    compliance = float(cp.sum(E_voxels * ce))
    dc = -PENAL * rho_phys ** (PENAL - 1) * (E0 - EMIN) * ce
    dc_f = sens_filter(dc)
    dv_f = sens_filter(dv)

    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l1 + l2)
        rho_new = cp.clip(rho * cp.sqrt(-dc_f / (dv_f * lmid)), rho - MOVE, rho + MOVE)
        rho_new = cp.clip(rho_new, 0.0, 1.0)
        if float(cp.mean(rho_new)) > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid

    change = float(cp.max(cp.abs(rho_new - rho)))
    rho = rho_new
    print(
        f"it {loop:3d}  c {compliance:.4e}  vol {float(cp.mean(rho_phys)):.3f}  "
        f"change {change:.3f}  cg {cg_iters}"
    )
    if change < CHANGE_TOL:
        break

print(f"optimization: {time.time() - tic:.2f}s")
rho_phys = density_filter(rho)
sol = u_gpu.get()

# --------------------------------------- export --------------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
density_field = mlhp.scalarFieldFromVoxelData(
    mlhp.DoubleVector(rho_phys.get().ravel("C")),
    nvoxels=NVOXELS,
    lengths=DOMAIN_LENGTHS,
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(density_field, "Density"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 1] * D)

out = str(RESULTS_DIR / f"topopt_mlhp_cuda_subvoxel_{D}D")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out}.pvtu")

# ----------------------------------- postprocessing ----------------------------------
if D == 2:
    fig, ax = plt.subplots()
    ax.imshow(
        rho_phys.get().reshape(NVOXELS).T,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
