import argparse
import time
from pathlib import Path

import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = args.dim

DEGREE = 2
SUB_VOXELS = 4
QUAD_ORDER = 1  # Gauss pts per direction per sub-cell (matches subvoxelV4)
NU = 0.3

# ------------------------------------ ct geometry ------------------------------------
ct_file = args.ct if args.ct else f"plate_hole_{D}D.npz"
ct = np.load(DATA_DIR / ct_file)
indicator = ct["indicator"]  # uint8

if D == 2:
    Lx, Ly = float(ct["Lx"]), float(ct["Ly"])
    Nx, Ny = indicator.shape
    ncells = [Nx, Ny]
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

assert all(n % SUB_VOXELS == 0 for n in ncells), (
    f"grid dims {ncells} must all be divisible by SUB_VOXELS={SUB_VOXELS}"
)

macro_elem_lengths = [l * SUB_VOXELS for l in elem_lengths]

nu_field = mlhp.scalarField(D, NU)

# ----------------------------- local macro element + basis ---------------------------
# One macro element (containing SUB_VOXELS^D sub-voxels) with the same basis used per
# element in the global problem.  K_locals[s] is the stiffness contribution of sub-voxel
# s alone (E=1); the CUDA kernel later forms K_e = sum_s E_s * K_locals[s].
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


# --------------------- single-pass preintegration (sub-cell bucketing) ---------------
# A full-element grid quadrature already places QUAD_ORDER^D points in every sub-cell.
# Evaluate the strain matrix B once at all points, then bucket each B^T C B w into the
# sub-cell its point falls in: K_locals[s] = sum_{q in V_s} B_q^T C B_q w_q.  Total
# quadrature work equals one full-element integration -- no symmetry reconstruction, and
# K_locals feeds cuda_assemble_K_e directly as K_refs[s, ndof_e^2].

# constitutive matrix C with E = 1 (Voigt, matching mlhp's plane-stress / isotropic)
if D == 2:  # plane stress, [eps_xx, eps_yy, gamma_xy]
    C = np.array([[1.0, NU, 0.0],
                  [NU, 1.0, 0.0],
                  [0.0, 0.0, (1.0 - NU) / 2.0]]) / (1.0 - NU**2)
    n_strain = 3
else:  # isotropic, [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]
    lam = NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = 1.0 / (2.0 * (1.0 + NU))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    n_strain = 6

# quadrature points + physical weights for the single macro element (cell 0)
quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS] * D)
parts = quadrature.evaluate(mesh_local, 0, [QUAD_ORDER] * D)
xyz = np.array([p for part in parts for p in part.xyz])  # (n_q, D)
w = np.array([wq for part in parts for wq in part.weights])  # (n_q,)
w *= np.prod(macro_elem_lengths) / w.sum()  # fold in the constant affine det(J)
n_q = len(xyz)

# sub-cell index per point (C-order: matches scalarFieldFromVoxelData and the CUDA kernel)
sub_len = np.array(macro_elem_lengths) / SUB_VOXELS
ijk = np.minimum((xyz / sub_len).astype(int), SUB_VOXELS - 1)
s_of_q = np.ravel_multi_index(ijk.T, [SUB_VOXELS] * D)

# strain-displacement matrix B at every point: B[q, strain, dof].  For each vector DOF i
# we read the full difforder-1 Jacobian J[c, j] = d(u_c)/dx_j of the field it activates and
# map it to a Voigt strain column.  Indexing B columns by the actual DOF i (rather than an
# assumed node/component interleaving) matches basis_local's own DOF numbering exactly.
B = np.zeros((n_q, n_strain, ndof_e))
for i in range(ndof_e):
    dofs = mlhp.DoubleVector([0.0] * ndof_e)
    dofs[i] = 1.0
    evaluator = mlhp.vectorEvaluator(basis_local, dofs, difforder=1)
    J = np.array([evaluator(p) for p in xyz]).reshape(n_q, D, D)
    if D == 2:
        B[:, 0, i] = J[:, 0, 0]  # eps_xx
        B[:, 1, i] = J[:, 1, 1]  # eps_yy
        B[:, 2, i] = J[:, 0, 1] + J[:, 1, 0]  # gamma_xy
    else:
        B[:, 0, i] = J[:, 0, 0]  # eps_xx
        B[:, 1, i] = J[:, 1, 1]  # eps_yy
        B[:, 2, i] = J[:, 2, 2]  # eps_zz
        B[:, 3, i] = J[:, 1, 2] + J[:, 2, 1]  # gamma_yz
        B[:, 4, i] = J[:, 0, 2] + J[:, 2, 0]  # gamma_xz
        B[:, 5, i] = J[:, 0, 1] + J[:, 1, 0]  # gamma_xy

# single pass: scatter every point's B^T C B w into its sub-cell's matrix
tic = time.time()
BtCB = np.einsum("qmi,mn,qnj->qij", B, C, B) * w[:, None, None]
K_locals = np.zeros((n_sub, ndof_e, ndof_e))
np.add.at(K_locals, s_of_q, BtCB)
print(f"preintegration (1 pass, {n_sub} subvoxels): {time.time() - tic:.2f}s")

# ------------------------------------ verification -----------------------------------
# per sub-voxel: every bucket must match mlhp's own masked integration
rng = np.random.default_rng(0)
err = max(
    np.max(np.abs(K_locals[s] - get_K_local_rep(s)))
    for s in rng.integers(0, n_sub, 12)
)
print(f"max |bucket - get_K_local_rep| over 12 random sub-voxels: {err:.3e}")

# global: summed buckets must equal the macro element integrated over all sub-voxels
K_full = sum(get_K_local_rep(s) for s in range(n_sub))
print(f"|sum_s K_locals - full element|: {np.max(np.abs(K_locals.sum(0) - K_full)):.3e}")
