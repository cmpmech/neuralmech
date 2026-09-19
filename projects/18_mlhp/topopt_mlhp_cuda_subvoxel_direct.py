import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.ndimage as cnd
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
args = parser.parse_args()

# -------------------------------- optimization settings ------------------------------
D = args.dim

DEGREE = 1
SUB_VOXELS = 8  # 6  # design (voxel) resolution per coarse finite element
QUAD_ORDER = DEGREE + 1

VOLFRAC = 0.5
PENAL = 3.0
# RMIN = 1.5  # 2  # voxels
RMIN = 2
MOVE = 0.2
MAX_ITER = 100
CHANGE_TOL = 0.1  # TODO or 0.01

E0 = 1.0
EMIN = 1e-6
NU = 0.3
TRACTION = 1.0

DTYPE = cp.float64  # cp.float32 for single precision
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64

BLOCK = 1024

# half-MBB beam (left symmetry), 3:1 aspect; voxels divisible by SUB_VOXELS
if D == 2:
    nvoxels = [360, 120]
    # nvoxels = [90, 30]
    domain_lengths = [3.0, 1.0]
else:
    nvoxels = [96, 32, 32]
    domain_lengths = [3.0, 1.0, 1.0]

assert all(n % SUB_VOXELS == 0 for n in nvoxels)
nelems = [n // SUB_VOXELS for n in nvoxels]
elem_lengths = [l / n for l, n in zip(domain_lengths, nelems)]
n_voxels = int(np.prod(nvoxels))

nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems, lengths=domain_lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
ndof = basis.ndof()
print(basis)

# -------------------------------- boundary conditions --------------------------------
# left face: x-symmetry (u_x = 0); bottom-right corner/edge: roller (u_y = 0);
# 3D additionally fixes u_z on the z-min face to remove out-of-plane rigid motion.
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

# ------------------------ local preintegrated stiffness matrices ---------------------
# one reference matrix per subvoxel position inside a coarse element
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
)  # (n_sub, ndof_e, ndof_e)
print(f"preintegration ({n_sub} subvoxels): {time.time() - tic:.2f}s")

# ----------------------------------- load vector -------------------------------------
# downward line/point load over a one-element-wide patch at the top-left corner
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

# interior-ordered (reduced) load vector consumed directly by the direct solver;
# copy out of the mlhp vector (vector.array is a view) before releasing it
interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
interior_idx = np.where(interior_mask)[0]
rhs_red = np.array(vector.array, dtype=np_dtype)
del vector

efts = np.array(basis.locationMaps())

# --------------------------------------- cuda ----------------------------------------
cuda_source = (BASE_DIR / "../../solvers/kernels/mlhp_kernels.cu").read_text()
cuda_options = ("-DUSE_FLOAT",) if DTYPE == cp.float32 else ()
module = cp.RawModule(code=cuda_source, options=cuda_options)
assemble_K_e_kernel = module.get_function("assemble_K_e_density_kernel")
sensitivity_kernel = module.get_function("compliance_sensitivity_kernel")

# indexing parameters for the per-element kernels (1 as dummy for 2D)
N_elems = int(np.prod(nelems))
Ny_elem, Nz_elem = nelems[1], nelems[2] if D == 3 else 1
Ny_vox, Nz_vox = nvoxels[1], nvoxels[2] if D == 3 else 1
Sz = SUB_VOXELS if D == 3 else 1
grid = (N_elems + BLOCK - 1) // BLOCK

K_refs_gpu = cp.array(K_refs.ravel("C"), dtype=DTYPE)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)

K_e_gpu = cp.empty(N_elems * ndof_e * ndof_e, dtype=DTYPE)

# fixed COO sparsity pattern of the global stiffness, matching the row-major
# K_e[i*ndof_e + j] layout; only the values change across optimization iterations
rows = np.repeat(efts, ndof_e, axis=1).ravel()  # local i index (slow)
cols = np.tile(efts, (1, ndof_e)).ravel()  # local j index (fast)

# -------------------------------------- filter ---------------------------------------
# conic density filter of radius RMIN (voxel units) with edge normalization Hs
ceil_r = int(np.ceil(RMIN))
axes = np.meshgrid(*([np.arange(-ceil_r, ceil_r + 1)] * D), indexing="ij")
h = np.maximum(0.0, RMIN - np.sqrt(sum(a**2 for a in axes)))
h_gpu = cp.array(h, dtype=DTYPE)
Hs_gpu = cnd.convolve(cp.ones(nvoxels, dtype=DTYPE), h_gpu, mode="constant")

density_filter = lambda x: cnd.convolve(x, h_gpu, mode="constant") / Hs_gpu
sens_filter = lambda g: cnd.convolve(g / Hs_gpu, h_gpu, mode="constant")


def solve(E_voxels):
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
    # assemble the global sparse matrix on the host and reduce to interior dofs
    K_e_host = cp.asnumpy(K_e_gpu).astype(np_dtype, copy=False)
    K_full = sp.coo_matrix((K_e_host, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K_red = K_full[interior_idx][:, interior_idx]

    u_red = pypardiso.spsolve(K_red, rhs_red)
    u = np.zeros(ndof, dtype=np_dtype)
    u[interior_idx] = u_red
    return u


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
    return ce.reshape(nvoxels)


# ----------------------------------- optimization ------------------------------------
rho = cp.full(nvoxels, VOLFRAC, dtype=DTYPE)
dv = cp.ones(nvoxels, dtype=DTYPE)

tic = time.time()
for loop in range(1, MAX_ITER + 1):
    rho_phys = density_filter(rho)
    E_voxels = EMIN + rho_phys**PENAL * (E0 - EMIN)

    t_solve = time.time()
    u = solve(E_voxels)
    t_solve = time.time() - t_solve
    ce = sensitivity(cp.array(u, dtype=DTYPE))

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
        f"change {change:.3f}  solve {t_solve:.2f}s"
    )
    if change < CHANGE_TOL:
        break

print(f"optimization: {time.time() - tic:.2f}s")
rho_phys = density_filter(rho)
sol = u

# --------------------------------------- export --------------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
density_field = mlhp.scalarFieldFromVoxelData(
    mlhp.DoubleVector(rho_phys.get().ravel("C")),
    nvoxels=nvoxels,
    lengths=domain_lengths,
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(density_field, "Density"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 1] * D)

out = str(RESULTS_DIR / f"topopt_mlhp_cuda_subvoxel_direct_{D}D")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out}.pvtu")

# ----------------------------------- postprocessing ----------------------------------
if D == 2:
    fig, ax = plt.subplots()
    ax.imshow(
        rho_phys.get().reshape(nvoxels).T,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()
