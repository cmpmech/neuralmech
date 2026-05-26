import argparse
import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg
import mlhp
import numpy as np


def _k_refs_worker(args):
    s_start, s_end, D, DEGREE, n_sub, S, macro_lengths, NU = args
    import mlhp
    import numpy as np

    macro_lengths = list(macro_lengths)
    kinematics = mlhp.smallStrainKinematics(D)
    nu_field = mlhp.scalarField(D, NU)
    mesh_ref = mlhp.makeRefinedGrid(
        mlhp.makeGrid(ncells=[1] * D, lengths=macro_lengths)
    )
    basis_ref = mlhp.makeHpTrunkSpace(mesh_ref, degree=DEGREE, nfields=D)
    de = mlhp.combineDirichletDofs([])
    quadrature = mlhp.gridQuadrature(nsubcells=[S] * D)
    ind_s = np.zeros(n_sub, dtype=np.float32)
    results = []
    for s in range(s_start, s_end):
        ind_s[:] = 0.0
        ind_s[s] = 1.0
        E_field_s = mlhp.scalarFieldFromVoxelData(
            mlhp.FloatVector(ind_s.tolist()), nvoxels=[S] * D, lengths=macro_lengths
        )
        c_s = (
            mlhp.planeStressMaterial(E_field_s, nu_field)
            if D == 2
            else mlhp.isotropicElasticMaterial(E_field_s, nu_field)
        )
        i_s = mlhp.staticDomainIntegrand(
            kinematics, c_s, mlhp.vectorField(D, [0.0] * D)
        )
        m_s = mlhp.allocateSparseMatrix(basis_ref, de[0])
        v_s = mlhp.allocateRhsVector(m_s)
        mlhp.integrateOnDomain(
            basis_ref, i_s, [m_s, v_s], quadrature=quadrature, dirichletDofs=de
        )
        results.append(np.array(m_s.todense()))
    return s_start, results


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)
parser.add_argument(
    "--fine",
    action="store_true",
    help="export at voxel resolution (one VTU cell per voxel)",
)
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = args.dim

DEGREE = 3
ALPHA = 1e-5
SUB_VOXELS = 8

DTYPE = cp.float64
np_dtype = np.float64
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
    f"grid dims {ncells} must all be divisible by SUB_VOXELS={SUB_VOXELS}"
)

ncells_elem = [n // SUB_VOXELS for n in ncells]
macro_lengths = [SUB_VOXELS * h for h in elem_lengths]

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

# ---------------------- subvoxel preintegrated reference matrices ---------------------
# K_refs[s] = stiffness contribution from subvoxel s only (E=1, all others 0).
# Indexed in C-order: 2D s = sx*S+sy, 3D s = sx*S^2+sy*S+sz  (same as indicator.ravel("C")).
kinematics = mlhp.smallStrainKinematics(D)
mesh_ref = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=macro_lengths))
basis_ref = mlhp.makeHpTrunkSpace(mesh_ref, degree=DEGREE, nfields=D)
ndof_e = basis_ref.ndof()
assert ndof_e <= 256, (
    f"ndof_e={ndof_e} > MAX_NDOF_E=256; raise the limit in mlhp_kernels.cu"
)
de = mlhp.combineDirichletDofs([])

n_sub = SUB_VOXELS**D
print(f"n_sub={n_sub}, ndof_e={ndof_e}")

cache_key = (
    f"D{D}_p{DEGREE}_S{SUB_VOXELS}_"
    + "_".join(f"{x:.8e}" for x in macro_lengths)
    + f"_nu{NU:.8e}"
)
cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
cache_file = BASE_DIR / f".K_refs_{cache_hash}.npy"

if cache_file.exists():
    K_refs = np.load(cache_file)
    print(f"K_refs loaded from cache: {cache_file.name}")
else:
    K_refs = np.zeros((n_sub, ndof_e, ndof_e))
    n_workers = min(mp.cpu_count(), n_sub)
    batch_size = max(1, (n_sub + n_workers - 1) // n_workers)
    batches = [
        (
            i,
            min(i + batch_size, n_sub),
            D,
            DEGREE,
            n_sub,
            SUB_VOXELS,
            tuple(macro_lengths),
            NU,
        )
        for i in range(0, n_sub, batch_size)
    ]
    tic = time.time()
    completed = 0
    with mp.Pool(processes=n_workers) as pool:
        for s_start, batch_K in pool.imap_unordered(_k_refs_worker, batches):
            for i, K_s in enumerate(batch_K):
                K_refs[s_start + i] = K_s
            completed += len(batch_K)
            print(f"  K_refs {completed}/{n_sub} ({time.time() - tic:.1f}s)")
    np.save(cache_file, K_refs)
    print(f"K_refs ({n_sub}): {time.time() - tic:.2f}s — cached to {cache_file.name}")

# # -------------------------------------- assembly -------------------------------------
# c_rhs = (
#     mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
#     if D == 2
#     else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
# )
# i_rhs = mlhp.staticDomainIntegrand(kinematics, c_rhs, mlhp.vectorField(D, [0.0] * D))
# matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
# vector = mlhp.allocateRhsVector(matrix)

# tic = time.time()
# mlhp.integrateOnDomain(
#     basis,
#     i_rhs,
#     [matrix, vector],
#     quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
#     dirichletDofs=dirichlet,
# )
# traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
# neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
# right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
# mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
# print(f"assembly: {time.time() - tic:.2f}s")

# interior_mask = np.ones(ndof, dtype=bool)
# interior_mask[constrained_dofs] = False
# rhs = np.zeros(ndof)
# rhs[np.where(interior_mask)[0]] = np.array(list(vector))
# del matrix, vector

# efts = np.array(basis.locationMaps())

# # --------------------------------------- cuda ----------------------------------------
# # Compile to SASS (not PTX) so first kernel launch has no JIT overhead.
# cc = cp.cuda.Device().compute_capability
# cuda_source = (BASE_DIR / "mlhp_kernels.cu").read_text()
# cuda_options = (f"-arch=sm_{cc}",)
# if DTYPE == cp.float32:
#     cuda_options += ("-DUSE_FLOAT",)
# tic = time.time()
# module = cp.RawModule(code=cuda_source, options=cuda_options, backend="nvcc")
# kernel_assemble_K_e = module.get_function("cuda_assemble_K_e")
# kernel_matvec_elem = module.get_function("cuda_matvec_elem")
# kernel_k_diag_elem = module.get_function("cuda_k_diag_elem")
# print(f"CUDA compile (sm_{cc}): {time.time() - tic:.2f}s")

# n_elem = int(np.prod(ncells_elem))
# grid = (n_elem + BLOCK - 1) // BLOCK

# E_scalar = np_dtype(E)
# alpha_scalar = np_dtype(ALPHA)
# Ny_elem = ncells_elem[1]
# Nz_elem = ncells_elem[2] if D == 3 else 1
# Ny_vox = ncells[1]
# Nz_vox = ncells[2] if D == 3 else 1
# Sz = SUB_VOXELS if D == 3 else 1

# cp.cuda.Stream.null.synchronize()
# tic = time.time()
# K_refs_gpu = cp.array(K_refs.ravel("C"), dtype=DTYPE)
# efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
# indicator_gpu = cp.array(indicator.ravel("C"), dtype=cp.uint8)
# rhs_gpu = cp.array(rhs, dtype=DTYPE)
# constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)
# cp.cuda.Stream.null.synchronize()
# print(f"GPU upload: {time.time() - tic:.3f}s")

# K_e_gpu = cp.empty(n_elem * ndof_e * ndof_e, dtype=DTYPE)
# cp.cuda.Stream.null.synchronize()
# tic = time.time()
# kernel_assemble_K_e(
#     (grid,),
#     (BLOCK,),
#     (
#         K_e_gpu,
#         indicator_gpu,
#         K_refs_gpu,
#         E_scalar,
#         alpha_scalar,
#         n_elem,
#         ndof_e,
#         n_sub,
#         SUB_VOXELS,
#         Sz,
#         Ny_elem,
#         Nz_elem,
#         Ny_vox,
#         Nz_vox,
#     ),
# )
# cp.cuda.Stream.null.synchronize()
# print(f"K_e assembly: {time.time() - tic:.3f}s")

# K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
# cp.cuda.Stream.null.synchronize()
# tic = time.time()
# kernel_k_diag_elem(
#     (grid,),
#     (BLOCK,),
#     (K_diag_gpu, efts_gpu, K_e_gpu, n_elem, ndof_e),
# )
# K_diag_gpu[constrained_gpu] = 1.0
# cp.cuda.Stream.null.synchronize()
# print(f"K_diag: {time.time() - tic:.3f}s")


# def matvec_gpu(u_gpu):
#     Ku_gpu = cp.zeros(ndof, dtype=DTYPE)
#     kernel_matvec_elem(
#         (grid,),
#         (BLOCK,),
#         (u_gpu, Ku_gpu, efts_gpu, K_e_gpu, n_elem, ndof_e),
#     )
#     Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
#     return Ku_gpu


# A_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=matvec_gpu)
# P_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

# # --------------------------------------- solve ---------------------------------------
# iters = [0]


# def callback(x):
#     iters[0] += 1


# cp.cuda.Stream.null.synchronize()
# tic = time.time()
# sol_gpu, info = cp_splinalg.cg(
#     A_op, rhs_gpu, M=P_op, tol=1e-10, maxiter=20000, callback=callback
# )
# cp.cuda.Stream.null.synchronize()
# print(f"CG: {iters[0]} iterations, info={info}, {time.time() - tic:.2f}s")

# sol = sol_gpu.get()
# print(f"max displacement: {np.max(np.abs(sol)):.3e}")

# # --------------------------------------- export --------------------------------------
# all_dofs = mlhp.DoubleVector(sol.tolist())
# indicator_field = mlhp.scalarFieldFromVoxelData(
#     mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
#     nvoxels=ncells,
#     lengths=lengths,
# )
# processors = [
#     mlhp.solutionProcessor(D, all_dofs, "Displacement"),
#     mlhp.functionProcessor(indicator_field, "Indicator"),
# ]
# # --fine: one VTU cell per voxel so material distribution is visible per-voxel
# postmesh_res = SUB_VOXELS if args.fine else DEGREE + 2
# postmesh = mlhp.gridCellMesh([postmesh_res] * D)

# suffix = f"_vox" if args.fine else ""
# out = str(RESULTS_DIR / f"elastic_mlhp_cuda_sub{SUB_VOXELS}{suffix}_{ct_file[:-4]}")
# Path(out).parent.mkdir(parents=True, exist_ok=True)
# output = mlhp.PVtuOutput(filename=out)
# mlhp.basisOutput(basis, postmesh, output, processors)
# print(f"VTU written to {out}.pvtu")
