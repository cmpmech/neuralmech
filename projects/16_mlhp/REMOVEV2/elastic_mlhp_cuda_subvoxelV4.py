import argparse
import time
from itertools import combinations_with_replacement, product
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

DEGREE = 2
ALPHA = 1e-4
SUB_VOXELS = 8  # 4
QUAD_ORDER = DEGREE + 1  # Gauss pts per direction per sub-cell; default (DEGREE+1)
QUAD_ORDER = 1

DTYPE = cp.float64  # cp.float32 for single precision
np_dtype = np.float32 if DTYPE == cp.float32 else np.float64
CG_TOL = 1e-6 if DTYPE == cp.float32 else 1e-10

BLOCK = 1024
PRECOMPILED = False
# compiler_options = ()
compiler_options = ("--use_fast_math", "--gpu-architecture=compute_120")

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

ncells_macro = [n // SUB_VOXELS for n in ncells]
macro_elem_lengths = [l * SUB_VOXELS for l in elem_lengths]

nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells_macro, lengths=lengths))
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
# K_locals[u] = stiffness contribution from one sub-voxel only (E=1, others 0).
# We only integrate the symmetry representatives: fold every coordinate into the lower
# half (reflections) and sort them (axis swaps), i.e. 0 <= i <= j (<= k) < SUB_VOXELS/2.
kinematics = mlhp.smallStrainKinematics(D)
mesh_local = mlhp.makeRefinedGrid(
    mlhp.makeGrid(ncells=[1] * D, lengths=macro_elem_lengths)
)
basis_local = mlhp.makeHpTrunkSpace(mesh_local, degree=DEGREE, nfields=D)
ndof_e = basis_local.ndof()
n_sub = SUB_VOXELS**D


def get_K_local_rep(s):
    indicator_s = np.zeros(n_sub, dtype=np.float32)
    indicator_s[s] = 1.0
    E_s = mlhp.scalarFieldFromVoxelData(
        mlhp.FloatVector(indicator_s.tolist()),
        nvoxels=[SUB_VOXELS] * D,
        lengths=macro_elem_lengths,
    )
    constitutive_s = (
        mlhp.planeStressMaterial(E_s, nu_field)
        if D == 2
        else mlhp.isotropicElasticMaterial(E_s, nu_field)
    )
    integrand_s = mlhp.staticDomainIntegrand(
        kinematics, constitutive_s, mlhp.vectorField(D, [0.0] * D)
    )
    # integrateOnDomain accumulates; no reset API → allocate fresh each iteration
    matrix_s = mlhp.allocateSparseMatrix(basis_local)
    rhs_s = mlhp.allocateRhsVector(matrix_s)
    mlhp.integrateOnDomain(
        basis_local,
        integrand_s,
        [matrix_s, rhs_s],
        quadrature=mlhp.gridQuadrature(nsubcells=[SUB_VOXELS] * D),
        orderDeterminor=mlhp.absoluteQuadratureOrder([QUAD_ORDER] * D),
    )
    return np.array(matrix_s.todense())


# sorted coordinate tuples 0 <= i <= j (<= k) < SUB_VOXELS/2 -> one per symmetry orbit.
# count = multiset coefficient: 2D (h)(h+1)/2, 3D (h)(h+1)(h+2)/6 with h = SUB_VOXELS/2.
reps = list(combinations_with_replacement(range(SUB_VOXELS // 2), D))
n_unique_sub = len(reps)
K_local_reps = np.zeros((n_unique_sub, ndof_e, ndof_e), dtype=np.float64)

tic = time.time()
for u, coords in enumerate(reps):
    s = int(np.ravel_multi_index(coords, [SUB_VOXELS] * D))
    K_local_reps[u] = get_K_local_rep(s)
print(f"preintegration ({n_unique_sub} / {n_sub} subvoxels): {time.time() - tic:.2f}s")


# ------------------------------------ reverse map ------------------------------------
# For every voxel s, find its representative slot in K_locals and the symmetry that maps
# the representative onto s:
#   flips[s]     = which axes are mirrored (coordinate sat in the upper half),
#   axis_perm[s] = axis order that sorts the folded coordinates (the axis swaps).
# A mirror is coord -> S-1-coord; an axis swap is the diagonal mirror; a mirror+swap is a
# rotation.  The element stiffness then follows as K[s] = P^T K_locals[u] P, with P the
# signed DOF permutation built from these mirror/swap generators (see subvoxelV3).
rep_to_u = {coords: u for u, coords in enumerate(reps)}

rep_index = np.zeros(n_sub, dtype=np.int32)
flips = np.zeros((n_sub, D), dtype=bool)
perms = np.zeros((n_sub, D), dtype=np.int32)

for s in range(n_sub):
    coords = np.array(np.unravel_index(s, [SUB_VOXELS] * D))  # in i, j, k
    folded = np.minimum(coords, SUB_VOXELS - 1 - coords)  # find closest edge (mirror)
    perm = np.argsort(folded, kind="stable")
    flips[s] = coords > SUB_VOXELS - 1 - coords
    perms[s] = perm
    rep_index[s] = rep_to_u[tuple(folded[perm])]

    # # sanity: mirror + un-sort the representative reproduces the original voxel
    # recovered = folded[perm][np.argsort(perm)]
    # recovered = np.where(flips[s], SUB_VOXELS - 1 - recovered, recovered)
    # assert np.array_equal(recovered, coords)

print(f"reverse map: {n_sub} voxels -> {n_unique_sub} representatives")

print(flips)
print(perms)
print(rep_index)


# # ----------------------------- signed DOF permutations -------------------------------
# # Each cell symmetry sends every basis function to +/- another one.  Sampling the local
# # basis at probe points before and after the symmetry reveals that signed permutation
# # directly, so we never hand-derive the DOF shuffle.  (Same construction as subvoxelV3.)
# probe = [list(p) for p in product(np.linspace(0.13, 0.87, DEGREE + 3), repeat=D)]
# probe = [[t * h for t, h in zip(p, macro_elem_lengths)] for p in probe]


# def sample_basis(points):
#     phi = np.zeros((ndof_e, len(points), D))
#     for i in range(ndof_e):
#         dofs = mlhp.DoubleVector([0.0] * ndof_e)
#         dofs[i] = 1.0
#         evaluator = mlhp.vectorEvaluator(basis_local, dofs, difforder=0)
#         phi[i] = [evaluator(p) for p in points]
#     return phi


# def signed_perm(point_map, comp):
#     """Match each (transformed) basis function to +/- another to get the permutation."""
#     phi = sample_basis(probe)
#     image = comp(sample_basis([point_map(p) for p in probe]))
#     P = np.zeros((ndof_e, ndof_e))
#     for i in range(ndof_e):
#         for j in range(ndof_e):
#             if np.allclose(image[i], phi[j]):
#                 P[j, i] = 1.0
#             elif np.allclose(image[i], -phi[j]):
#                 P[j, i] = -1.0
#     assert np.allclose(np.abs(P).sum(0), 1), "symmetry is not a signed permutation"
#     return P


# def reflect_perm(d):  # mirror axis d: point flips, d-displacement flips sign
#     def comp(a):
#         a = a.copy()
#         a[..., d] *= -1.0
#         return a

#     pm = lambda p: [macro_elem_lengths[k] - x if k == d else x for k, x in enumerate(p)]
#     return signed_perm(pm, comp)


# def axis_perm_to_signed(perm):  # reorder axes by perm: points and components both permute
#     def comp(a):
#         return a[..., perm]

#     pm = lambda p: [p[perm[k]] for k in range(D)]
#     return signed_perm(pm, comp)


# # One reflection generator per axis, one swap matrix per possible axis ordering (D! of them).
# P_reflect = [reflect_perm(d) for d in range(D)]
# P_axis = {p: axis_perm_to_signed(list(p)) for p in product(range(D), repeat=D) if sorted(p) == list(range(D))}


# # --------------------------- reconstruct K[s] from its orbit -------------------------
# # K[s] = M.T @ K_locals[rep_index[s]] @ M, with M the signed DOF permutation that folds
# # (reflections recorded in flips[s]) then sorts (the axis swap recorded in axis_perm[s]).
# def reconstruct_K_local(s):
#     M = np.eye(ndof_e)
#     for d in range(D):  # reflections fold the voxel into the lower half
#         if flips[s, d]:
#             M = P_reflect[d] @ M
#     M = P_axis[tuple(axis_perm[s])] @ M  # axis swap sorts the folded coordinates
#     return M.T @ K_locals[rep_index[s]] @ M


# # sanity: reconstruction must match the brute-force per-voxel integration
# rng = np.random.default_rng(0)
# err = max(
#     np.max(np.abs(reconstruct_K_local(s) - get_K_local(s)))
#     for s in rng.integers(0, n_sub, 12)
# )
# print(f"max |reconstructed - brute| over 12 random sub-voxels: {err:.3e}")
