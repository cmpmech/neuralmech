import argparse
import time
from itertools import product
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cp_splinalg
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse as sp

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../../data"
RESULTS_DIR = BASE_DIR / "../../../results/3D"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--smooth", type=int, default=3)
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = args.dim
DEGREE = args.degree  # degree the forward problem is solved at; the cycle runs at p=1

SUB_VOXELS = 8  # fine-element subvoxel resolution per direction
NLEVELS_H = 3  # number of p=1 (h-)multigrid levels
NU_PRE = args.smooth  # Chebyshev smoother degree (matvecs per pre-/post-smooth)
NU_POST = args.smooth
POWER_ITERS = 20  # power iterations for the per-level max-eigenvalue estimate
EIG_RATIO = 20.0  # smoothing band [cheb_b/EIG_RATIO, cheb_b]
EIG_SAFETY = 1.1  # over-estimate factor on lambda_max

DTYPE = cp.float64  # cp.float32 for single precision
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
CG_TOL = 1e-6 if DTYPE == cp.float32 else 1e-10

BLOCK = 1024

E = 210.0
NU = 0.3
ALPHA = 1e-4
FORCE = 1.0

# ------------------------------------ ct geometry ------------------------------------
ct_file = args.ct if args.ct else f"plate_hole_{D}D.npz"
ct = np.load(DATA_DIR / ct_file)
indicator = ct["indicator"]  # uint8

if D == 2:
    Lx, Ly = float(ct["Lx"]), float(ct["Ly"])
    nvoxels = list(indicator.shape)
    domain_lengths = [Lx, Ly]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    nvoxels = list(indicator.shape)
    domain_lengths = [Lx, Ly, Lz]

coarsen = 2 ** (NLEVELS_H - 1)
assert all(n % (SUB_VOXELS * coarsen) == 0 for n in nvoxels), (
    f"grid {nvoxels} must be divisible by SUB_VOXELS*2^(NLEVELS_H-1)={SUB_VOXELS * coarsen}"
)

nelems0 = [n // SUB_VOXELS for n in nvoxels]
nu_field = mlhp.scalarField(D, NU)
kinematics = mlhp.smallStrainKinematics(D)
material = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)

# per-voxel material stiffness, for coarse-level coefficient averaging
stiff_vox = E * np.maximum(indicator.astype(np.float64) / 255.0, ALPHA)

# --------------------------------------- cuda ----------------------------------------
cuda_source = (BASE_DIR / "../../../solvers/kernels/mlhp_kernels.cu").read_text()
cuda_options = ("-DUSE_FLOAT",) if DTYPE == cp.float32 else ()
module = cp.RawModule(code=cuda_source, options=cuda_options)
assemble_subvoxel = module.get_function("assemble_K_e_kernel")
assemble_density = module.get_function("assemble_K_e_density_kernel")
Ku_kernel = module.get_function("Ku_subvoxel_kernel")
K_diag_kernel = module.get_function("K_diag_subvoxel_kernel")

indicator_gpu = cp.array(indicator.ravel("C"), dtype=cp.uint8)


# ------------------------------- vertex / dof bookkeeping ----------------------------
# local positions (within an element field block) of the 2^D corner/vertex modes,
# detected empirically per degree from the cardinal shape functions of one element.
def vertex_local_positions(degree):
    mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=[1.0] * D))
    basis = mlhp.makeHpTrunkSpace(mesh, degree=degree, nfields=1)
    nd = basis.ndof()
    corners = list(product([0.0, 1.0], repeat=D))
    pts = [np.array([c[d] for c in corners]) for d in range(D)]
    vpos = {}
    for j in range(nd):
        u = np.zeros(nd)
        u[j] = 1.0
        vals = np.array(mlhp.scalarEvaluator(basis, mlhp.DoubleVector(u))(*pts))
        hot = np.abs(vals) > 0.5
        if hot.sum() == 1 and np.allclose(vals, hot.astype(float)):
            vpos[int(np.argmax(hot))] = j
    return [vpos[lc] for lc in range(2**D)]


VPOS = {deg: vertex_local_positions(deg) for deg in {DEGREE, 1}}
CORNERS = list(product([0, 1], repeat=D))


def vertex_dofmap(basis, nelems, degree):
    efts = np.array(basis.locationMaps())
    lpf = efts.shape[1] // D  # local dofs per field block
    vpos = VPOS[degree]
    Ny = nelems[1]
    Nz = nelems[2] if D == 3 else 1
    dofmap = np.full(tuple(n + 1 for n in nelems) + (D,), -1, dtype=np.int64)
    for e in range(efts.shape[0]):
        if D == 2:
            base = (e // Ny, e % Ny)
        else:
            base = (e // (Ny * Nz), (e // Nz) % Ny, e % Nz)
        for lc, corner in enumerate(CORNERS):
            node = tuple(base[d] + corner[d] for d in range(D))
            for f in range(D):
                dofmap[node + (f,)] = efts[e][f * lpf + vpos[lc]]
    assert dofmap.min() >= 0
    return dofmap


# ----------------------------------- level assembly ----------------------------------
def boundary_dofs(basis):
    faces = [0, 2] if D == 2 else [0, 2, 4]
    bc = [
        mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [f], ifield=f // 2)
        for f in faces
    ]
    return mlhp.combineDirichletDofs(bc)


def preintegrate(degree, elem_lengths, nsubcells):
    mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
    basis = mlhp.makeHpTrunkSpace(mesh, degree=degree, nfields=D)
    integrand = mlhp.staticDomainIntegrand(
        kinematics, material, mlhp.vectorField(D, [0.0] * D)
    )
    quad = mlhp.gridQuadrature(nsubcells=[nsubcells] * D)
    K_refs = mlhp.integratePartitionMatrices(
        basis, integrand, quad, mlhp.absoluteQuadratureOrder([degree + 1] * D)
    )
    return K_refs, basis.ndof()


def make_level(degree, nelems, subvoxel):
    mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems, lengths=domain_lengths))
    basis = mlhp.makeHpTrunkSpace(mesh, degree=degree, nfields=D)
    ndof = basis.ndof()
    dirichlet = boundary_dofs(basis)
    constrained = np.array(dirichlet[0])

    efts = np.array(basis.locationMaps())
    efts_gpu = cp.array(efts.ravel("C"), dtype=cp.int32)
    constrained_gpu = cp.array(constrained, dtype=cp.int32)
    elem_lengths = [domain_lengths[d] / nelems[d] for d in range(D)]

    N_elems = int(np.prod(nelems))
    Ny_elem = nelems[1]
    Nz_elem = nelems[2] if D == 3 else 1
    grid = (N_elems + BLOCK - 1) // BLOCK

    if subvoxel:
        S = nvoxels[0] // nelems[0]
        Sz = S if D == 3 else 1
        n_sub = S**D
        K_refs, ndof_e = preintegrate(degree, elem_lengths, S)
        K_refs_gpu = cp.array(K_refs.ravel("C"), dtype=DTYPE)
        K_e_gpu = cp.zeros(N_elems * ndof_e * ndof_e, dtype=DTYPE)
        Ny_vox = nvoxels[1]
        Nz_vox = nvoxels[2] if D == 3 else 1
        assemble_subvoxel(
            (grid,),
            (BLOCK,),
            (
                K_e_gpu,
                indicator_gpu,
                K_refs_gpu,
                np_dtype(E),
                np_dtype(ALPHA),
                N_elems,
                ndof_e,
                n_sub,
                S,
                Sz,
                Ny_elem,
                Nz_elem,
                Ny_vox,
                Nz_vox,
            ),
        )
    else:
        K_refs, ndof_e = preintegrate(degree, elem_lengths, 1)
        K_refs_gpu = cp.array(K_refs.ravel("C"), dtype=DTYPE)
        K_e_gpu = cp.zeros(N_elems * ndof_e * ndof_e, dtype=DTYPE)
        # effective per-element coefficient = average voxel stiffness over the element
        f = [nvoxels[d] // nelems[d] for d in range(D)]
        if D == 2:
            E_eff = stiff_vox.reshape(nelems[0], f[0], nelems[1], f[1]).mean((1, 3))
        else:
            E_eff = stiff_vox.reshape(
                nelems[0], f[0], nelems[1], f[1], nelems[2], f[2]
            ).mean((1, 3, 5))
        E_eff_gpu = cp.array(E_eff.ravel("C"), dtype=DTYPE)
        assemble_density(
            (grid,),
            (BLOCK,),
            (
                K_e_gpu,
                E_eff_gpu,
                K_refs_gpu,
                N_elems,
                ndof_e,
                1,
                1,
                1,
                Ny_elem,
                Nz_elem,
                Ny_elem,
                Nz_elem,
            ),
        )

    K_diag_gpu = cp.zeros(ndof, dtype=DTYPE)
    K_diag_kernel((grid,), (BLOCK,), (K_diag_gpu, efts_gpu, K_e_gpu, N_elems, ndof_e))
    K_diag_gpu[constrained_gpu] = 1.0

    mask = cp.ones(ndof, dtype=DTYPE)  # 1 on free dofs, 0 on constrained
    mask[constrained_gpu] = 0.0

    return {
        "degree": degree,
        "nelems": nelems,
        "ndof": ndof,
        "ndof_e": ndof_e,
        "basis": basis,
        "dirichlet": dirichlet,
        "efts_gpu": efts_gpu,
        "constrained_gpu": constrained_gpu,
        "K_e_gpu": K_e_gpu,
        "K_diag_gpu": K_diag_gpu,
        "N_elems": N_elems,
        "grid": grid,
        "mask": mask,
        # preallocated V-cycle work buffers (reused every iteration)
        "Ku": cp.empty(ndof, dtype=DTYPE),
        "x": cp.empty(ndof, dtype=DTYPE),
        "r": cp.empty(ndof, dtype=DTYPE),
        "d": cp.empty(ndof, dtype=DTYPE),
    }


# ------------------------------------- hierarchy -------------------------------------
tic = time.time()
levels = []
if DEGREE > 1:
    levels.append(make_level(DEGREE, nelems0, subvoxel=True))  # solve operator
    levels.append(make_level(1, nelems0, subvoxel=True))  # p->1 on same mesh
    h_start = 1
else:
    levels.append(make_level(1, nelems0, subvoxel=True))
    h_start = 0

for k in range(1, NLEVELS_H):
    nel = [n // (2**k) for n in nelems0]
    levels.append(make_level(1, nel, subvoxel=False))
L = len(levels) - 1
print(f"hierarchy ({len(levels)} levels): {[lv['ndof'] for lv in levels]}")
print(f"assembly: {time.time() - tic:.2f}s")


# ----------------------------- inter-grid transfer operators -------------------------
def build_injection(level_fine, level_coarse):
    dm_f = vertex_dofmap(
        level_fine["basis"], level_fine["nelems"], level_fine["degree"]
    )
    dm_c = vertex_dofmap(level_coarse["basis"], level_coarse["nelems"], 1)
    rows = cp.array(dm_f.ravel(), dtype=cp.int32)
    cols = cp.array(dm_c.ravel(), dtype=cp.int32)
    data = cp.ones(rows.size, dtype=DTYPE)
    return cusp.csr_matrix(
        (data, (rows, cols)), shape=(level_fine["ndof"], level_coarse["ndof"])
    )


def axis_contrib(I):
    if I % 2 == 0:
        return [(I // 2, 1.0)]
    return [((I - 1) // 2, 0.5), ((I + 1) // 2, 0.5)]


def build_h_prolong(level_fine, level_coarse):
    dm_f = vertex_dofmap(level_fine["basis"], level_fine["nelems"], 1)
    dm_c = vertex_dofmap(level_coarse["basis"], level_coarse["nelems"], 1)
    rows, cols, data = [], [], []
    fine_shape = tuple(n + 1 for n in level_fine["nelems"])
    for node in np.ndindex(fine_shape):
        for combo in product(*[axis_contrib(node[d]) for d in range(D)]):
            cnode = tuple(c[0] for c in combo)
            w = float(np.prod([c[1] for c in combo]))
            for fld in range(D):
                rows.append(int(dm_f[node + (fld,)]))
                cols.append(int(dm_c[cnode + (fld,)]))
                data.append(w)
    return cusp.csr_matrix(
        (
            cp.array(data, dtype=DTYPE),
            (cp.array(rows, dtype=cp.int32), cp.array(cols, dtype=cp.int32)),
        ),
        shape=(level_fine["ndof"], level_coarse["ndof"]),
    )


def validate(P, level_fine, level_coarse):
    uc = np.random.RandomState(0).rand(level_coarse["ndof"])
    field = mlhp.vectorEvaluator(level_coarse["basis"], mlhp.DoubleVector(uc))
    ref = np.array(mlhp.interpolateNodes(level_fine["basis"], field))
    got = (P @ cp.array(uc, dtype=DTYPE)).get()
    return float(np.max(np.abs(got - ref)))


tic = time.time()
P = []  # P[i] prolongs level i+1 -> level i
R = []
for i in range(L):
    if DEGREE > 1 and i == 0:
        Pi = build_injection(levels[0], levels[1])
    else:
        Pi = build_h_prolong(levels[i], levels[i + 1])
        err = validate(Pi, levels[i], levels[i + 1])
        assert err < 1e-9, f"prolongation {i}->{i + 1} failed validation: {err:.2e}"
    P.append(Pi)
    R.append(Pi.T.tocsr())
print(f"transfer operators: {time.time() - tic:.2f}s")


# ------------------------------------- multigrid -------------------------------------
# fused elementwise kernels: one launch each, writing into preallocated buffers.
# identity rows on constrained dofs: o = a*mask + u*(1-mask)
apply_idrows = cp.ElementwiseKernel(
    "T a, T mask, T u", "T o", "o = a * mask + u * (T(1) - mask)", "apply_idrows"
)
# masked residual r = (b - Ku) * mask
masked_sub = cp.ElementwiseKernel(
    "T b, T Ku, T mask", "T r", "r = (b - Ku) * mask", "masked_sub"
)
# masked correction x = (x + y) * mask
masked_add = cp.ElementwiseKernel(
    "T x, T y, T mask", "T o", "o = (x + y) * mask", "masked_add"
)
# Chebyshev first half-step: r=(b-Ku)*mask; d=Dinv*r*inv_theta; x=(x+d)*mask
cheb_first = cp.ElementwiseKernel(
    "T b, T Ku, T Dinv, T mask, float64 inv_theta, T x_in",
    "T r, T d, T x_out",
    """
    r = (b - Ku) * mask;
    d = Dinv * r * inv_theta;
    x_out = (x_in + d) * mask;
    """,
    "cheb_first",
)
# Chebyshev recurrence step: d=c1*d+c2*Dinv*(b-Ku)*mask; x=(x+d)*mask
cheb_next = cp.ElementwiseKernel(
    "T b, T Ku, T Dinv, T mask, float64 c1, float64 c2, T x_in, T d_in",
    "T d_out, T x_out",
    """
    T rr = (b - Ku) * mask;
    d_out = c1 * d_in + c2 * (Dinv * rr);
    x_out = (x_in + d_out) * mask;
    """,
    "cheb_next",
)


def matvec(level, u, out):
    out.fill(0.0)
    Ku_kernel(
        (level["grid"],),
        (BLOCK,),
        (
            u,
            out,
            level["efts_gpu"],
            level["K_e_gpu"],
            level["N_elems"],
            level["ndof_e"],
        ),
    )
    apply_idrows(out, level["mask"], u, out)
    return out


# per-level diagonal inverse and Chebyshev smoothing band (top of D^-1 A spectrum)
def estimate_lmax(level):
    v = cp.asarray(np.random.RandomState(0).rand(level["ndof"]), dtype=DTYPE)
    v *= level["mask"]
    v /= cp.linalg.norm(v)
    lam = 1.0
    for _ in range(POWER_ITERS):
        w = level["Dinv"] * matvec(level, v, level["Ku"])
        w *= level["mask"]
        lam = float(cp.linalg.norm(w))
        v = w / lam
    return lam


for lv in levels:
    lv["Dinv"] = 1.0 / lv["K_diag_gpu"]
    lv["cheb_b"] = EIG_SAFETY * estimate_lmax(lv)
    lv["cheb_a"] = lv["cheb_b"] / EIG_RATIO


# Chebyshev-accelerated Jacobi smoother (damps the high end of D^-1 A), in-place on x
def smooth(level, b, x, deg):
    Ku, r, d, mask, Dinv = (
        level["Ku"],
        level["r"],
        level["d"],
        level["mask"],
        level["Dinv"],
    )
    theta = 0.5 * (level["cheb_b"] + level["cheb_a"])
    delta = 0.5 * (level["cheb_b"] - level["cheb_a"])
    sigma = theta / delta
    rho = 1.0 / sigma
    matvec(level, x, Ku)
    cheb_first(b, Ku, Dinv, mask, 1.0 / theta, x, r, d, x)
    for _ in range(deg - 1):
        rho_new = 1.0 / (2.0 * sigma - rho)
        matvec(level, x, Ku)
        cheb_next(b, Ku, Dinv, mask, rho * rho_new, 2.0 * rho_new / delta, x, d, d, x)
        rho = rho_new
    return x


# dense inverse of the (small) coarsest operator, precomputed once on the GPU so
# the coarse solve inside every V-cycle is a single gemv with no host transfer
def build_coarse_inverse():
    lvl = levels[L]
    ne, nd_e, ndof = lvl["N_elems"], lvl["ndof_e"], lvl["ndof"]
    efts = lvl["efts_gpu"].get().reshape(ne, nd_e).astype(np.int64)
    Ke = lvl["K_e_gpu"].get().reshape(ne, nd_e, nd_e)
    rows = np.repeat(efts[:, :, None], nd_e, axis=2).ravel()
    cols = np.repeat(efts[:, None, :], nd_e, axis=1).ravel()
    Kd = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof)).toarray()
    ic = lvl["constrained_gpu"].get()
    Kd[ic, :] = 0.0
    Kd[:, ic] = 0.0
    Kd[ic, ic] = 1.0
    return cp.asarray(np.linalg.inv(Kd), dtype=DTYPE)


coarse_Kinv = build_coarse_inverse()


def vcycle(i, b):
    level = levels[i]
    if i == L:
        return cp.dot(coarse_Kinv, b, out=level["x"])
    x = level["x"]
    x.fill(0.0)
    smooth(level, b, x, NU_PRE)
    matvec(level, x, level["Ku"])
    r = level["r"]
    masked_sub(b, level["Ku"], level["mask"], r)
    rc = R[i] @ r
    rc *= levels[i + 1]["mask"]
    ec = vcycle(i + 1, rc)
    masked_add(x, P[i] @ ec, level["mask"], x)
    smooth(level, b, x, NU_POST)
    return x


# ------------------------------------- load vector -----------------------------------
fine = levels[0]
matrix = mlhp.allocateSparseMatrix(fine["basis"], fine["dirichlet"][0])
vector = mlhp.allocateRhsVector(matrix)
del matrix
traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(
    mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems0, lengths=domain_lengths)), [1]
)
mlhp.integrateOnSurface(
    fine["basis"], neumann, [vector], right_quad, dirichletDofs=fine["dirichlet"]
)
constrained0 = np.array(fine["dirichlet"][0])
interior_mask = np.ones(fine["ndof"], dtype=bool)
interior_mask[constrained0] = False
rhs = np.zeros(fine["ndof"])
rhs[np.where(interior_mask)[0]] = vector.array
rhs_gpu = cp.array(rhs, dtype=DTYPE)
del vector

# --------------------------------------- solve ---------------------------------------
A0_out = cp.empty(fine["ndof"], dtype=DTYPE)  # dedicated buffer for the CG operator
A0 = cp_splinalg.LinearOperator(
    (fine["ndof"],) * 2, matvec=lambda u: matvec(fine, u, A0_out)
)
M_jac = cp_splinalg.LinearOperator(
    (fine["ndof"],) * 2, matvec=lambda v: v * fine["Dinv"]
)
M_mg = cp_splinalg.LinearOperator((fine["ndof"],) * 2, matvec=lambda r: vcycle(0, r))

for name, precond in [("Jacobi-CG", M_jac), ("MGCG", M_mg)]:
    iters = [0]
    cp.cuda.Stream.null.synchronize()
    tic = time.time()
    sol, info = cp_splinalg.cg(
        A0,
        rhs_gpu,
        M=precond,
        rtol=CG_TOL,
        maxiter=20000,
        callback=lambda x: iters.__setitem__(0, iters[0] + 1),
    )
    cp.cuda.Stream.null.synchronize()
    print(f"{name:10s}: {iters[0]:5d} iters, info={info}, {time.time() - tic:.2f}s")
    if name == "MGCG":
        sol_mg = sol
    else:
        sol_jac = sol

print(f"max|u_mg - u_jac|: {float(cp.max(cp.abs(sol_mg - sol_jac))):.2e}")

# --------------------------------------- export --------------------------------------
sol = sol_mg.get()
all_dofs = mlhp.DoubleVector(sol.tolist())
processors = [mlhp.solutionProcessor(D, all_dofs, "Displacement")]
postmesh = mlhp.gridCellMesh([DEGREE + 1] * D)
out = str(RESULTS_DIR / f"elastic_mlhp_cuda_subvoxel_multigrid_{D}D_p{DEGREE}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(fine["basis"], postmesh, output, processors)
print(f"VTU written to {out}.pvtu")
