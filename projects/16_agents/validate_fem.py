import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

# -------------------------------------- settings -------------------------------------
# reference FEM ground truth (adapted from 16_mlhp/voxelized/elastic_cuda.py); the
# load axis is x, equivalent to the method's z-tension because the geometry is
# fully symmetric; used only for validation, never for calibrating the method
RESOLUTION = 64
DEGREE = 1
E = 1.0
NU = 0.3
FORCE = 1.0
CG_TOL = 1e-10
BLOCK = 1024

# -------------------------------------- load data ------------------------------------
ct = np.load(DATA_DIR / f"void_cube_{RESOLUTION}.npz")
indicator = ct["indicator"]
lengths = [float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])]
nelems = list(indicator.shape)
elem_lengths = [lengths[i] / nelems[i] for i in range(3)]

keep_mask = indicator.ravel("C") > 0
n_elems = int(keep_mask.sum())
indicator_data = indicator.ravel("C")[keep_mask]

# ---------------------------------------- mesh ---------------------------------------
base_grid = mlhp.makeFilteredGrid(
    mlhp.makeGrid(ncells=nelems, lengths=lengths), mask=keep_mask
)
mesh = mlhp.makeRefinedGrid(base_grid)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=3)
ndof = basis.ndof()

bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(3, 0.0), basis, [face], ifield=face // 2
    )
    for face in [0, 2, 4]
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained_dofs = np.array(dirichlet[0])

kinematics = mlhp.smallStrainKinematics(3)
constitutive = mlhp.isotropicElasticMaterial(
    mlhp.scalarField(3, E), mlhp.scalarField(3, NU)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, constitutive, mlhp.vectorField(3, [0.0] * 3)
)

mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * 3, lengths=elem_lengths))
basis_local = mlhp.makeHpTrunkSpace(mesh_local, degree=DEGREE, nfields=3)
matrix_local = mlhp.allocateSparseMatrix(basis_local)
rhs_local = mlhp.allocateRhsVector(matrix_local)
mlhp.integrateOnDomain(
    basis_local,
    integrand,
    [matrix_local, rhs_local],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * 3),
)
K_local = np.array(matrix_local.todense())

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
del matrix

traction = FORCE / (lengths[1] * lengths[2])
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(3, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)

interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained_dofs] = False
rhs = np.zeros(ndof)
rhs[interior_mask] = vector.array
del vector

efts = np.array(basis.locationMaps())

# --------------------------------------- solve ---------------------------------------
cuda_source = (BASE_DIR / "../../solvers/kernels/mlhp_kernels.cu").read_text()
module = cp.RawModule(code=cuda_source)
Ku_kernel = module.get_function("Ku_kernel")
K_diag_kernel = module.get_function("K_diag_kernel")

ndof_e = K_local.shape[0]
grid = (n_elems + BLOCK - 1) // BLOCK
K_local_gpu = cp.array(K_local.ravel("C"), dtype=cp.float64)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
indicator_gpu = cp.array(indicator_data, dtype=cp.uint8)
rhs_gpu = cp.array(rhs, dtype=cp.float64)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)

K_diag_gpu = cp.zeros(ndof, dtype=cp.float64)
K_diag_kernel(
    (grid,),
    (BLOCK,),
    (K_diag_gpu, indicator_gpu, efts_gpu, K_local_gpu, np.float64(E),
     np.float64(0.0), n_elems, ndof_e),
)
K_diag_gpu[constrained_gpu] = 1.0


def get_Ku(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=cp.float64)
    Ku_kernel(
        (grid,),
        (BLOCK,),
        (u_gpu, Ku_gpu, indicator_gpu, efts_gpu, K_local_gpu, np.float64(E),
         np.float64(0.0), n_elems, ndof_e),
    )
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu


KU_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=get_Ku)
K_diag_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

tic = time.perf_counter()
sol_gpu, info = cp_splinalg.cg(KU_op, rhs_gpu, M=K_diag_op, rtol=CG_TOL, maxiter=40000)
cp.cuda.Stream.null.synchronize()
toc_solve = time.perf_counter() - tic
sol = sol_gpu.get()
print(f"fem solve: info={info}, {toc_solve:.2f} s, max |u| {np.abs(sol).max():.4e}")

# ----------------------------------- postprocessing ----------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
processors = [
    mlhp.solutionProcessor(3, all_dofs, "Displacement"),
    mlhp.stressProcessor(all_dofs, kinematics, constitutive, "Stress"),
]
postmesh = mlhp.gridCellMesh([1] * 3)
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, processors)
points = np.array(result.mesh().points()).reshape(-1, 3)
displacement = np.array(result.data()[0]).reshape(-1, 3)
stress = np.array(result.data()[1]).reshape(-1, 9)

# stiffness from the mean normal displacement of the loaded face
loaded = points[:, 0] > lengths[0] - 1e-9
k0_fem = FORCE / displacement[loaded, 0].mean()

# stress concentration along the equatorial ligament: shell averages, because the
# voxel staircase produces singular corner spikes that make a raw max meaningless
h = lengths[0] / RESOLUTION
radius = 0.2
r_yz = np.sqrt((points[:, 1] - 0.5) ** 2 + (points[:, 2] - 0.5) ** 2)
plane = np.abs(points[:, 0] - 0.5) < h
sxx_nominal = FORCE / (lengths[1] * lengths[2])

nu_g = NU
c3 = (4.0 - 5.0 * nu_g) / (2.0 * (7.0 - 5.0 * nu_g))
c5 = 9.0 / (2.0 * (7.0 - 5.0 * nu_g))
goodier_surface = (27.0 - 15.0 * nu_g) / (2.0 * (7.0 - 5.0 * nu_g))

shell_r, shell_kt, shell_goodier = [], [], []
for k in range(2, 9):
    shell = plane & (np.abs(r_yz - (radius + k * h)) < 0.5 * h)
    r_mean = r_yz[shell].mean()
    shell_r.append(r_mean)
    shell_kt.append(stress[shell, 0].mean() / sxx_nominal)
    shell_goodier.append(1.0 + c3 * (radius / r_mean) ** 3 + c5 * (radius / r_mean) ** 5)
shell_r = np.array(shell_r)
shell_kt = np.array(shell_kt)
shell_goodier = np.array(shell_goodier)

# surface value by scaling the analytic profile shape to the measured shells
excess_scale = ((shell_kt - 1.0) / (shell_goodier - 1.0)).mean()
kt_fem = 1.0 + excess_scale * (goodier_surface - 1.0)
print(f"fem k0 {k0_fem:.4f}")
print(f"fem ligament kt shells {np.round(shell_kt, 4)}")
print(f"goodier shells         {np.round(shell_goodier, 4)}")
print(f"fem surface kt (extrapolated) {kt_fem:.4f} (goodier {goodier_surface:.4f})")

f_init_fem = 1.0 / kt_fem
print(f"fem initiation load {f_init_fem:.4f}")

# ------------------------------------- comparison ------------------------------------
method = RESULTS_DIR / "fracture_percolation.npz"
if method.exists():
    m = np.load(method)
    print("\nmethod vs fem reference:")
    print(f"  k0        {float(m['k0']):.4f} vs {k0_fem:.4f} "
          f"({100 * (float(m['k0']) / k0_fem - 1):+.1f}%)")
    amp_method = float(m["amp"].max())
    print(f"  amp       {amp_method:.4f} vs {kt_fem:.4f} "
          f"({100 * (amp_method / kt_fem - 1):+.1f}%)")
    f_init = float(m["f"][1])
    print(f"  f_init    {f_init:.4f} vs {f_init_fem:.4f} "
          f"({100 * (f_init / f_init_fem - 1):+.1f}%)")
    print(f"  fem solve {toc_solve:.1f} s (one linear solve, no fracture)")
