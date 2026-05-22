import argparse
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg
import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

D = args.dim

E = 210.0
nu = 0.3
force = 1.0

# ----------------------------- CT geometry ----------------------------------
ct_default = BASE_DIR.parent.parent / "data" / f"plate_hole_{D}D.npz"
ct = np.load(args.ct or ct_default)
indicator = ct["indicator"]
Lx = float(ct["Lx"])
Ly = float(ct["Ly"])

if D == 2:
    Nx, Ny = indicator.shape
    ncells = [Nx, Ny]
    lengths = [Lx, Ly]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lz = float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

E_values = (E * indicator).ravel("C")

E_vec = mlhp.DoubleVector(E_values)
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=ncells, lengths=lengths)
nu_field = mlhp.scalarField(D, nu)

# ------------------------------- mesh + basis --------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells, lengths=lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=args.degree, nfields=D)
ndof = basis.ndof()
print(basis)

# Uni-axial tension: each surface constrains its own normal component only.
# face 0 (x-) → ux=0,  face 2 (y-) → uy=0,  face 4 (z-) → uz=0
# Tangential directions are free → Poisson contraction allowed on all faces.
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2
    )
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained = np.array(dirichlet[0])

# ----------------------------- K_ref (1-element) ----------------------------
kinematics = mlhp.smallStrainKinematics(D)
mesh1 = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis1 = mlhp.makeHpTrunkSpace(mesh1, degree=args.degree, nfields=D)
c_ref = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
i_ref = mlhp.staticDomainIntegrand(kinematics, c_ref, mlhp.vectorField(D, [0.0] * D))
de = mlhp.combineDirichletDofs([])
m_ref = mlhp.allocateSparseMatrix(basis1, de[0])
v_ref = mlhp.allocateRhsVector(m_ref)
mlhp.integrateOnDomain(
    basis1,
    i_ref,
    [m_ref, v_ref],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
    dirichletDofs=de,
)
K_ref = np.array(m_ref.todense())

# ----------------------------- RHS assembly ---------------------------------
c_full = (
    mlhp.planeStressMaterial(E_field, nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(E_field, nu_field)
)
i_full = mlhp.staticDomainIntegrand(kinematics, c_full, mlhp.vectorField(D, [0.0] * D))
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

tic = time.time()
mlhp.integrateOnDomain(
    basis,
    i_full,
    [matrix, vector],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
    dirichletDofs=dirichlet,
)
traction = force / Ly if D == 2 else force / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
print(f"RHS assembly: {time.time() - tic:.2f}s")

interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = np.array(list(vector))
del matrix, vector

efts = np.array(basis.locationMaps())  # (n_elem, ndof_e)

# ----------------------------- CUDA setup -----------------------------------
cuda_source = (BASE_DIR / "elasticity_mf_mlhp.cu").read_text()
module = cp.RawModule(code=cuda_source)
kernel_matvec = module.get_function("cuda_matvec")
kernel_k_diag = module.get_function("cuda_k_diag")

n_elem = len(E_values)
ndof_e = K_ref.shape[0]
block = 256
grid = (n_elem + block - 1) // block

tic = time.time()
K_ref_gpu = cp.array(K_ref.ravel("C"), dtype=cp.float64)
efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
E_values_gpu = cp.array(E_values, dtype=cp.float64)
rhs_gpu = cp.array(rhs, dtype=cp.float64)
constrained_gpu = cp.array(constrained, dtype=cp.int32)
print(f"GPU upload: {time.time() - tic:.3f}s")

# diagonal preconditioner on GPU
K_diag_gpu = cp.zeros(ndof, dtype=cp.float64)
kernel_k_diag(
    (grid,), (block,), (K_diag_gpu, E_values_gpu, efts_gpu, K_ref_gpu, n_elem, ndof_e)
)
K_diag_gpu[constrained_gpu] = 1.0


def matvec_gpu(u_gpu):
    Ku_gpu = cp.zeros(ndof, dtype=cp.float64)
    kernel_matvec(
        (grid,),
        (block,),
        (u_gpu, Ku_gpu, E_values_gpu, efts_gpu, K_ref_gpu, n_elem, ndof_e),
    )
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu


A_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=matvec_gpu)
P_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

# --------------------------------- solve ------------------------------------
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

# ----------------------------- postprocessing --------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.DoubleVector(indicator.ravel("C")), nvoxels=ncells, lengths=lengths
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([args.degree + 2] * D)

if D == 2:
    result = mlhp.DataAccumulator()
    mlhp.basisOutput(basis, postmesh, result, processors)
    disp = np.array(result.data()[0])
    ux = disp[0::2]

    tri = result.triangulation()
    ind_viz = np.array(result.data()[1])
    tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)

    fig, ax = plt.subplots()
    cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
    fig.colorbar(cb)
    ax.set_aspect("equal")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.minorticks_off()
    fig.tight_layout(pad=0)

    if args.book:
        out = BASE_DIR.parent.parent / "results" / "16_elastic_fem_mlhp_cuda_ux.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    elif args.animate:
        out = (
            BASE_DIR.parent.parent
            / "results"
            / "animations"
            / "16_elastic_fem_mlhp_cuda_ux.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    else:
        plt.show()
else:
    out_stem = str(BASE_DIR / "output" / "elastic_mlhp_cuda_3d")
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    output = mlhp.PVtuOutput(filename=out_stem)
    mlhp.basisOutput(basis, postmesh, output, processors)
    print(f"VTU written to {out_stem}.pvtu")
