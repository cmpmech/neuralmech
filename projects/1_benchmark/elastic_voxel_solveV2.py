import argparse
import json
import time
from pathlib import Path

import cupy as cp
import cupyx
import cupyx.scipy.sparse.linalg as cp_splinalg
import mlhp
import numpy as np
from pyevtk.hl import imageToVTK

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data/abc"
DEF_DIR = DATA_DIR / "elasticity/fixture_definition"
FIXTURE_DIR = DATA_DIR / "elasticity/fixture"
SOLUTION_DIR = DATA_DIR / "elasticity/voxel_solution"
RESULTS_DIR = BASE_DIR / "../../results/abc/elasticity/voxel_solution"
KERNEL_FILE = BASE_DIR / "../../solvers/kernels/mlhp_kernels.cu"

D = 3

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--case", type=int, default=1)
args = parser.parse_args()

# ----------------------------------- solver settings ---------------------------------
DEGREE = 1
ALPHA = 1e-4  # void stiffness floor for cut voxels (filtered ones never reach the kernel)
FILTER_THRESHOLD = 0  # drop voxels with indicator <= this from the mesh

PRECOND = "block"  # "block" = nodal 3x3 block-Jacobi, "diag" = scalar diagonal (V1)

DTYPE = cp.float32
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
CG_TOL = 1e-6 if DTYPE == cp.float32 else 1e-10
MAXITER = 20000
BLOCK = 1024

EXPORT_VTU = True  # full-mesh .pvtu (displacement + indicator) for ParaView
EXPORT_VOXEL = True  # sample solution onto the voxel grid -> .npz
EXPORT_VOXEL_VTU = True  # write the voxelized solution as .vti

name = f"{args.geometry:09}_abc"

# corner-offset table for a degree-1 element: local node l sits at the grid vertex
# (ix + l>>2&1, iy + l>>1&1, iz + l&1) -- verified against mlhp's locationMaps ordering.
OFFSET = np.array([[(l >> 2) & 1, (l >> 1) & 1, l & 1] for l in range(8)])


# ----------------------------------- helpers -----------------------------------------
def sample_field(field, points, chunk=200000):
    out = np.empty((len(points), field.odim))
    for i in range(0, len(points), chunk):
        p = points[i : i + chunk]
        out[i : i + chunk] = np.array(field(p[:, 0], p[:, 1], p[:, 2])).reshape(
            -1, field.odim
        )
    return out


def von_mises(stress):  # stress: (N, 9) row-major 3x3 Cauchy tensor
    s = stress.reshape(-1, 3, 3)
    sxx, syy, szz = s[:, 0, 0], s[:, 1, 1], s[:, 2, 2]
    sxy, syz, sxz = s[:, 0, 1], s[:, 1, 2], s[:, 0, 2]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )


# ----------------------------------- geometry & fixture ------------------------------
spec = json.loads((DEF_DIR / f"{name}_{args.case}.json").read_text())
E = spec["material"]["E"]
NU = spec["material"]["nu"]

fix = np.load(FIXTURE_DIR / f"{name}_{args.case}.npz")
indicator = fix["indicator"]  # uint8, (Nx, Ny, Nz)
dirichlet_mask = fix["dirichlet_mask"].astype(bool)  # (Nx, Ny, Nz, 3) on the outer hull
dirichlet_value = fix["dirichlet_value"]  # (Nx, Ny, Nz, 3)
neumann = fix["neumann"]  # (Nx, Ny, Nz, 3) traction vector on the outer hull
origin = np.asarray(fix["origin"])
spacing = float(fix["spacing"])
lengths = [float(fix["Lx"]), float(fix["Ly"]), float(fix["Lz"])]

ncells = indicator.shape
solid = indicator > FILTER_THRESHOLD  # kept elements
keep_mask = solid.ravel("C")
n_elem = int(keep_mask.sum())
print(
    f"{name} case {args.case} ({spec['label']}): "
    f"{indicator.size} voxels, {n_elem} solid ({100 * n_elem / indicator.size:.1f}%)",
    flush=True,
)

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
# hull boundary voxels can be translated into nodal constraints and forces.
efts = np.array(basis.locationMaps()).reshape(n_elem, D, 8)
kept = np.array(np.unravel_index(np.flatnonzero(keep_mask), ncells, order="C")).T
corner = kept[:, None, :] + OFFSET[None, :, :]  # (n_elem, 8, 3) grid-vertex index

vertex_dof = np.full((*[n + 1 for n in ncells], D), -1, dtype=np.int64)
for c in range(D):
    vertex_dof[corner[..., 0], corner[..., 1], corner[..., 2], c] = efts[:, c, :]

# ----------------------------- dirichlet constraints ---------------------------------
# A hull voxel's corner that is shared with a solid element (vertex_dof >= 0) is a
# surface node; constrain it in every masked component.
cdofs = []
cvals = []
for c in range(D):
    vox = np.argwhere(dirichlet_mask[..., c])
    if not len(vox):
        continue
    cv = (vox[:, None, :] + OFFSET[None, :, :]).reshape(-1, 3)
    dof = vertex_dof[cv[:, 0], cv[:, 1], cv[:, 2], c]
    val = np.repeat(dirichlet_value[..., c][dirichlet_mask[..., c]], 8)
    good = dof >= 0
    cdofs.append(dof[good])
    cvals.append(val[good])

cdofs = np.concatenate(cdofs)
cvals = np.concatenate(cvals)
order = np.argsort(cdofs, kind="stable")
constrained_dofs, first = np.unique(cdofs[order], return_index=True)
constrained_values = cvals[order][first]

# ----------------------------- neumann nodal forces ----------------------------------
# Distribute each hull voxel's traction over the solid face(s) it covers: a constant
# traction on a degree-1 square face gives t * area / 4 at each of the 4 face nodes.
rhs = np.zeros(ndof)
face_area = spacing * spacing
neu_vox = np.argwhere(np.any(neumann != 0.0, axis=-1))
for axis in range(D):
    for side in (-1, 1):
        nodes = [l for l in range(8) if OFFSET[l, axis] == (side > 0)]
        nb = neu_vox.copy()
        nb[:, axis] += side
        inb = (nb[:, axis] >= 0) & (nb[:, axis] < ncells[axis])
        touch = np.zeros(len(neu_vox), dtype=bool)
        touch[inb] = solid[nb[inb, 0], nb[inb, 1], nb[inb, 2]]
        sel = neu_vox[touch]  # hull voxels whose (axis, side) face abuts solid
        traction = neumann[sel[:, 0], sel[:, 1], sel[:, 2], :]
        for l in nodes:
            cv = sel + OFFSET[l]
            dof = vertex_dof[cv[:, 0], cv[:, 1], cv[:, 2], :]
            for c in range(D):
                np.add.at(rhs, dof[:, c], traction[:, c] * face_area / 4.0)

rhs[constrained_dofs] = constrained_values  # homogeneous in the current pipeline
print(
    f"\t{len(constrained_dofs)} constrained dofs, "
    f"{int(np.count_nonzero(rhs))} loaded dofs",
    flush=True,
)

# ----------------------- local preintegrated stiffness matrix ------------------------
tic = time.time()
kinematics = mlhp.smallStrainKinematics(D)
material = mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), mlhp.scalarField(D, NU))
integrand = mlhp.staticDomainIntegrand(kinematics, material, mlhp.vectorField(D, [0.0] * D))

mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=[spacing] * D))
basis_local = mlhp.makeHpTrunkSpace(mesh_local, degree=DEGREE, nfields=D)
matrix_local = mlhp.allocateSparseMatrix(basis_local)
rhs_local = mlhp.allocateRhsVector(matrix_local)
mlhp.integrateOnDomain(
    basis_local, integrand, [matrix_local, rhs_local],
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


def get_Ku(u_gpu):
    # zero the constrained dofs of the input so the operator stays symmetric
    # (A = P K P + (I-P)); CG needs SPD. Only zeroing the output rows leaves the
    # constrained columns K_fc in place -> non-symmetric -> CG crawls.
    u_free = u_gpu.copy()
    u_free[constrained_gpu] = 0.0
    Ku_gpu = cp.zeros(ndof, dtype=DTYPE)
    Ku_kernel(
        (grid,), (BLOCK,),
        (u_free, Ku_gpu, indicator_gpu, efts_gpu, K_local_gpu, E_scalar, alpha_scalar, n_elem, ndof_e),
    )
    Ku_gpu[constrained_gpu] = u_gpu[constrained_gpu]
    return Ku_gpu


KU_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=get_Ku)

# --------------------------------- preconditioner ------------------------------------
# DOFs are component-blocked: dof = c * nnode + node (all x dofs, then all y, then z).
# The scalar diagonal preconditioner ignores the strong x/y/z coupling within a node;
# inverting the full 3x3 nodal block instead is the cheap matrix-free upgrade.
if PRECOND == "diag":
    K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
    K_diag_kernel(
        (grid,), (BLOCK,),
        (K_diag_gpu, indicator_gpu, efts_gpu, K_local_gpu, E_scalar, alpha_scalar, n_elem, ndof_e),
    )
    K_diag_gpu[constrained_gpu] = 1.0
    M_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=lambda v: v / K_diag_gpu)

elif PRECOND == "block":
    assert ndof % D == 0, "block-Jacobi expects D dofs per node"
    nnode = ndof // D
    nloc = ndof_e // D  # local nodes per element (8 for degree 1)
    efts_node = efts.reshape(n_elem, D, nloc)
    # component-blocked layout: component c of node n is global dof c*nnode + n, so the
    # x-component dof already *is* the node index in [0, nnode).
    assert np.array_equal(efts_node[:, 1, :], efts_node[:, 0, :] + nnode)
    assert np.array_equal(efts_node[:, 2, :], efts_node[:, 0, :] + 2 * nnode)

    node_gpu = cp.array(efts_node[:, 0, :].ravel(), dtype=cp.int64)
    Ei_gpu = E * cp.maximum(indicator_gpu.astype(DTYPE) / 255.0, np_dtype(ALPHA))  # (n_elem,)

    # scatter each element's per-node 3x3 sub-block into the global nodal blocks.
    blocks_flat = cp.zeros(nnode * D * D, dtype=DTYPE)
    for a in range(D):
        for b in range(D):
            kab = cp.array(
                np.diag(K_local[a * nloc:(a + 1) * nloc, b * nloc:(b + 1) * nloc]),
                dtype=DTYPE,
            )  # (nloc,) -- node-aligned entries of the (a, b) sub-block
            contrib = (Ei_gpu[:, None] * kab[None, :]).ravel()
            cupyx.scatter_add(blocks_flat, node_gpu * (D * D) + (a * D + b), contrib)
    blocks = blocks_flat.reshape(nnode, D, D)

    # constrained dofs get an identity row/col in the operator -> identity in the block.
    cn = cp.asarray(constrained_dofs)
    nodes_c = cn % nnode
    comp_c = cn // nnode
    blocks[nodes_c, comp_c, :] = 0.0
    blocks[nodes_c, :, comp_c] = 0.0
    blocks[nodes_c, comp_c, comp_c] = 1.0

    blocks_inv = cp.linalg.inv(blocks)  # (nnode, 3, 3)

    def apply_block(v):
        vn = v.reshape(D, nnode).T  # (nnode, D) node-grouped
        out = cp.einsum("nij,nj->ni", blocks_inv, vn)
        return out.T.ravel()  # back to component-blocked

    M_op = cp_splinalg.LinearOperator((ndof, ndof), matvec=apply_block)

else:
    raise ValueError(f"unknown PRECOND {PRECOND!r}")

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
    f"\tCG ({PRECOND}) {iters[0]} iters, info={info}, {time.time() - tic:.2f} s, "
    f"max |u| {np.max(np.abs(sol)):.3e}",
    flush=True,
)

# ----------------------------------- surface vtu -------------------------------------
if EXPORT_VTU:
    indicator_field = mlhp.scalarFieldFromVoxelData(
        mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
        nvoxels=list(ncells), lengths=lengths, origin=origin.tolist(), outside=0.0,
    )
    processors = [
        mlhp.solutionProcessor(D, mlhp.DoubleVector(sol.tolist()), "Displacement"),
        mlhp.functionProcessor(indicator_field, "Indicator"),
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = mlhp.PVtuOutput(filename=str(RESULTS_DIR / f"{name}_{args.case}"))
    mlhp.basisOutput(basis, mlhp.gridCellMesh([DEGREE + 1] * D), output, processors)
    print(f"\t-> {RESULTS_DIR / f'{name}_{args.case}'}.pvtu", flush=True)

# ----------------------------------- voxel solution ----------------------------------
if EXPORT_VOXEL or EXPORT_VOXEL_VTU:
    displacement = np.zeros((*ncells, D))
    displacement[kept[:, 0], kept[:, 1], kept[:, 2], :] = sol[efts].mean(axis=2)

    axes = [origin[d] + spacing * (np.arange(ncells[d]) + 0.5) for d in range(D)]
    points = np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])
    mask = solid.ravel()
    mat = mlhp.isotropicElasticMaterial(mlhp.scalarField(D, E), mlhp.scalarField(D, NU))
    mech = mlhp.mechanicalEvaluator(basis, mlhp.DoubleVector(sol.tolist()), kinematics, mat)
    stress = sample_field(mech.stress, points[mask])

    stress_full = np.zeros((points.shape[0], 9))
    vm_full = np.zeros(points.shape[0])
    stress_full[mask] = stress
    vm_full[mask] = von_mises(stress)
    stress_tensor = stress_full.reshape(*ncells, 3, 3)
    vonmises = vm_full.reshape(ncells)

if EXPORT_VOXEL:
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    out = SOLUTION_DIR / f"{name}_{args.case}.npz"
    np.savez(
        out,
        indicator=indicator,
        displacement=displacement,
        stress=stress_tensor,
        von_mises=vonmises,
        origin=origin,
        spacing=spacing,
        Lx=lengths[0],
        Ly=lengths[1],
        Lz=lengths[2],
    )
    print(f"\t-> {out}", flush=True)

if EXPORT_VOXEL_VTU:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}_{args.case}_voxel"
    cell_data = {
        "indicator": np.ascontiguousarray(indicator),
        "von_mises": np.ascontiguousarray(vonmises),
        "displacement": tuple(np.ascontiguousarray(displacement[..., i]) for i in range(D)),
    }
    imageToVTK(str(out), origin=tuple(origin), spacing=(spacing,) * D, cellData=cell_data)
    print(f"\t-> {out}.vti", flush=True)
