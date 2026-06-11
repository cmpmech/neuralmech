import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cp_splinalg
import mlhp
import numpy as np
from pyevtk.hl import imageToVTK

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results/abc/elasticity/solution_voxel").resolve()
KERNEL_FILE = BASE_DIR / "../../solvers/kernels/mlhp_kernels.cu"

D = 3
name = "000000000_abc"

# ----------------------------------- solver settings ---------------------------------
DEGREE = 1
ALPHA = 1e-4  # void stiffness floor (no voids here, but kept for kernel parity)
FILTER_THRESHOLD = 0

DTYPE = cp.float32
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
CG_TOL = 1e-6 if DTYPE == cp.float32 else 1e-10
MAXITER = 20000
BLOCK = 1024

# ----------------------------------- uniaxial tension --------------------------------
# Hardcoded textbook tension to debug the BC path without the fixture pipeline: roller
# (single component) on the three low faces + uniform traction on the +x face. The
# result is constant strain, so the closed form below is exact for linear elements.
E = 210.0
NU = 0.3
TRACTION = 1.0  # tensile traction on the +x face
AXIS = 0  # pull along x (the bar's long axis)

LENGTHS = [0.2, 0.05, 0.05]
ORIGIN = np.array([0.0, 0.0, 0.0])
NCELLS = (40, 10, 10)

# corner-offset table for a degree-1 element: local node l sits at the grid vertex
# (ix + l>>2&1, iy + l>>1&1, iz + l&1) -- verified against mlhp's locationMaps ordering.
OFFSET = np.array([[(l >> 2) & 1, (l >> 1) & 1, l & 1] for l in range(8)])

# ----------------------------------- geometry ----------------------------------------
indicator = np.full(NCELLS, 255, dtype=np.uint8)  # fully solid bar
origin = ORIGIN
lengths = LENGTHS
ncells = indicator.shape
nnodes = tuple(n + 1 for n in ncells)
spacing = lengths[0] / ncells[0]  # cubic voxels
solid = indicator > FILTER_THRESHOLD
keep_mask = solid.ravel("C")
n_elem = int(keep_mask.sum())
print(f"{name} uniaxial tension: {indicator.size} voxels, {n_elem} solid", flush=True)

# --------------------------------------- mesh ----------------------------------------
base_grid = mlhp.makeGrid(ncells=list(ncells), lengths=lengths, origin=origin.tolist())
base_grid = mlhp.makeFilteredGrid(base_grid, mask=keep_mask)
mesh = mlhp.makeRefinedGrid(base_grid)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
ndof = basis.ndof()
print(basis, flush=True)

# ----------------------------- voxel -> dof mapping ----------------------------------
# Each kept element row in locationMaps is component-blocked: reshape (n_elem, D, 8),
# entry [e, c, l] is the dof of node l, component c. Scatter it to a vertex grid so the
# nodal BC fields can be translated into nodal constraints and forces.
efts = np.array(basis.locationMaps()).reshape(n_elem, D, 8)
kept = np.array(np.unravel_index(np.flatnonzero(keep_mask), ncells, order="C")).T
corner = kept[:, None, :] + OFFSET[None, :, :]  # (n_elem, 8, 3) grid-vertex index

vertex_dof = np.full((*nnodes, D), -1, dtype=np.int64)
for c in range(D):
    vertex_dof[corner[..., 0], corner[..., 1], corner[..., 2], c] = efts[:, c, :]

# ----------------------------- boundary conditions -----------------------------------
# Roller on the three low faces: constrain u_x on x=0, u_y on y=0, u_z on z=0. Traction
# t in +x is stamped on the high +x face nodes and distributed by the loop below.
dirichlet_mask = np.zeros((*nnodes, D), dtype=bool)
dirichlet_value = np.zeros((*nnodes, D))
dirichlet_mask[0, :, :, 0] = True
dirichlet_mask[:, 0, :, 1] = True
dirichlet_mask[:, :, 0, 2] = True

neumann = np.zeros((*nnodes, D))
neumann[ncells[0], :, :, AXIS] = TRACTION

cdofs = []
cvals = []
for c in range(D):
    node = np.argwhere(dirichlet_mask[..., c])  # (k, 3) node indices
    if not len(node):
        continue
    dof = vertex_dof[node[:, 0], node[:, 1], node[:, 2], c]
    val = dirichlet_value[node[:, 0], node[:, 1], node[:, 2], c]
    good = dof >= 0
    cdofs.append(dof[good])
    cvals.append(val[good])

cdofs = np.concatenate(cdofs)
cvals = np.concatenate(cvals)
order = np.argsort(cdofs, kind="stable")
constrained_dofs, first = np.unique(cdofs[order], return_index=True)
constrained_values = cvals[order][first]

# A solid element face is exposed (boundary) iff its neighbor voxel is void; if all 4 of
# its corners carry a traction flag the face is loaded. A constant traction on a
# degree-1 square face gives t * area / 4 at each of the 4 face nodes.
rhs = np.zeros(ndof)
face_area = spacing * spacing
neu_node = np.any(neumann != 0.0, axis=-1)  # (Nx+1, Ny+1, Nz+1) nodal flag
for axis in range(D):
    for side in (0, 1):  # 0 = low face (offset 0), 1 = high face (offset 1)
        face_nodes = [l for l in range(8) if OFFSET[l, axis] == side]
        nb = kept.copy()
        nb[:, axis] += 1 if side == 1 else -1
        inb = (nb[:, axis] >= 0) & (nb[:, axis] < ncells[axis])
        neighbor_solid = np.zeros(len(kept), dtype=bool)
        neighbor_solid[inb] = solid[nb[inb, 0], nb[inb, 1], nb[inb, 2]]
        sel = kept[~neighbor_solid]  # solid voxels whose (axis, side) face is exposed
        fn = sel[:, None, :] + OFFSET[None, face_nodes, :]  # (m, 4, 3) face-node index
        loaded = neu_node[fn[..., 0], fn[..., 1], fn[..., 2]].all(axis=1)
        sel, fn = sel[loaded], fn[loaded]
        if not len(sel):
            continue
        traction = neumann[fn[..., 0], fn[..., 1], fn[..., 2], :].mean(axis=1)  # (m, 3)
        for j in range(len(face_nodes)):
            cv = fn[:, j, :]
            dof = vertex_dof[cv[:, 0], cv[:, 1], cv[:, 2], :]
            for c in range(D):
                np.add.at(rhs, dof[:, c], traction[:, c] * face_area / 4.0)

rhs[constrained_dofs] = constrained_values
print(
    f"\t{len(constrained_dofs)} constrained dofs, "
    f"{int(np.count_nonzero(rhs))} loaded dofs",
    flush=True,
)

# ----------------------- local preintegrated stiffness matrix ------------------------
tic = time.time()
kinematics = mlhp.smallStrainKinematics(D)
material = mlhp.isotropicElasticMaterial(
    mlhp.scalarField(D, 1.0), mlhp.scalarField(D, NU)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, material, mlhp.vectorField(D, [0.0] * D)
)

mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=[spacing] * D))
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
ndof_e = K_local.shape[0]
efts_flat = np.array(basis.locationMaps())
print(f"\tpreintegration {time.time() - tic:.2f} s", flush=True)

# --------------------------------------- cuda ----------------------------------------
module = cp.RawModule(
    code=KERNEL_FILE.read_text(),
    options=("-DUSE_FLOAT",) if DTYPE == cp.float32 else (),
)
Ku_kernel = module.get_function("Ku_kernel")
K_diag_kernel = module.get_function("K_diag_kernel")
grid = (n_elem + BLOCK - 1) // BLOCK

K_local_gpu = cp.array(K_local.ravel("C"), dtype=DTYPE)
efts_gpu = cp.array(efts_flat.ravel("C"), dtype=cp.int32)
indicator_gpu = cp.array(indicator.ravel("C")[keep_mask], dtype=cp.uint8)
rhs_gpu = cp.array(rhs, dtype=DTYPE)
constrained_gpu = cp.array(constrained_dofs, dtype=cp.int32)
E_scalar = np_dtype(E)
alpha_scalar = np_dtype(ALPHA)

K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
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
        n_elem,
        ndof_e,
    ),
)
K_diag_gpu[constrained_gpu] = 1.0


def get_Ku(u_gpu):
    # zero the constrained dofs of the input so the operator stays symmetric
    # (A = P K P + (I-P)); CG needs SPD. Only zeroing the output rows leaves the
    # constrained columns K_fc in place -> non-symmetric -> CG crawls.
    u_free = u_gpu.copy()
    u_free[constrained_gpu] = 0.0
    Ku_gpu = cp.zeros(ndof, dtype=DTYPE)
    Ku_kernel(
        (grid,),
        (BLOCK,),
        (
            u_free,
            Ku_gpu,
            indicator_gpu,
            efts_gpu,
            K_local_gpu,
            E_scalar,
            alpha_scalar,
            n_elem,
            ndof_e,
        ),
    )
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu


KU_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=get_Ku)
M_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

# --------------------------------------- solve ---------------------------------------
iters = [0]


def callback(x):
    iters[0] += 1


cp.cuda.Stream.null.synchronize()
tic = time.time()
sol_gpu, info = cp_splinalg.cg(
    KU_op, rhs_gpu, M=M_op, rtol=CG_TOL, maxiter=MAXITER, callback=callback
)
cp.cuda.Stream.null.synchronize()
sol = cp.asnumpy(sol_gpu).astype(np.float64)
print(
    f"\tCG {iters[0]} iters, info={info}, {time.time() - tic:.2f} s, "
    f"max |u| {np.max(np.abs(sol)):.3e}",
    flush=True,
)

# ----------------------------------- analytical check --------------------------------
displacement = np.zeros((*nnodes, D))
valid = vertex_dof[..., 0] >= 0
displacement[valid] = sol[vertex_dof[valid]]

axes = [origin[d] + spacing * np.arange(nnodes[d]) for d in range(D)]
grids = np.meshgrid(*axes, indexing="ij")
coeff = TRACTION / E
analytical = np.zeros((*nnodes, D))
analytical[..., 0] = coeff * grids[0]
analytical[..., 1] = -NU * coeff * grids[1]
analytical[..., 2] = -NU * coeff * grids[2]
error = displacement - analytical

for c, lbl in enumerate("xyz"):
    rel = np.linalg.norm(error[..., c]) / np.linalg.norm(analytical[..., c])
    print(
        f"\tu_{lbl}: max|err| {np.abs(error[..., c]).max():.3e}, rel L2 {rel:.3e}",
        flush=True,
    )

# ----------------------------------- export ------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = RESULTS_DIR / f"{name}_V2_voxel"
imageToVTK(
    str(out),
    origin=tuple(float(o) for o in origin),
    spacing=(float(spacing),) * D,
    cellData={"indicator": np.ascontiguousarray(indicator)},
    pointData={
        "displacement": tuple(np.ascontiguousarray(displacement[..., i]) for i in range(D)),
        "analytical": tuple(np.ascontiguousarray(analytical[..., i]) for i in range(D)),
        "error": tuple(np.ascontiguousarray(error[..., i]) for i in range(D)),
    },
)
print(f"\t-> {out}.vti", flush=True)
