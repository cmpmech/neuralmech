import argparse
import time

import cupy as cp
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.ndimage

from helper import PointMaterial, PointwiseProblem, QuadratureTables, VoxelMultigrid

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
parser.add_argument("--resolution", type=int, default=32)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

# geometry
LENGTHS = [2.0, 1.0, 1.0][:D]
CELLS = 4  # gyroid unit cells along x
THICKNESS = 0.4  # gyroid sheet half-thickness in level-set units
PLATE = 0.05  # solid end plates as a fraction of the length

# discretization
RESOLUTION = args.resolution  # voxels per unit length
DEGREE = 1
SUB_VOXELS = 1

# physics
E_SOLID, ALPHA, NU = 1.0, 1e-4, 0.3
COMPRESSION = (
    0.12  # prescribed shortening as a fraction of the length, the walls buckle beyond
)
STEPS = 6

# solver
NEWTON_TOL = 1e-8  # relative residual norm
MAX_NEWTON = 30
CG_TOL = 1e-3  # relative residual of each Newton correction

# compressible neo-Hookean in the solid (prm[2] = 1), linear elastic in the void
MATERIAL = r"""
// prm = [E, nu, finite]: compressible neo-Hookean if finite, else linear small strain
template <int D>
__device__ void material(const double* H, const double* prm, double* P, double* A)
{
    const double E = prm[0], nu = prm[1];
    const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), mu = 0.5 * E / (1.0 + nu);
    const int D2 = D * D;
    if (prm[2] == 0.0) {
        double tr = 0.0;
        for (int i = 0; i < D; ++i) tr += H[i * D + i];
        for (int i = 0; i < D; ++i)
            for (int J = 0; J < D; ++J) {
                P[i * D + J] = (i == J ? lam * tr : 0.0) + mu * (H[i * D + J] + H[J * D + i]);
                for (int k = 0; k < D; ++k)
                    for (int L = 0; L < D; ++L)
                        A[(i * D + J) * D2 + k * D + L] = lam * (i == J) * (k == L)
                            + mu * ((i == k) * (J == L) + (i == L) * (J == k));
            }
        return;
    }
    double F[D * D], G[D * D];  // G = F^-T
    for (int i = 0; i < D2; ++i) F[i] = H[i] + (i % (D + 1) == 0);
    double J;
    if (D == 2) {
        J = F[0] * F[3] - F[1] * F[2];
        G[0] = F[3] / J; G[1] = -F[2] / J; G[2] = -F[1] / J; G[3] = F[0] / J;
    } else {
        const double c00 = F[4] * F[8] - F[5] * F[7], c01 = F[5] * F[6] - F[3] * F[8];
        const double c02 = F[3] * F[7] - F[4] * F[6];
        J = F[0] * c00 + F[1] * c01 + F[2] * c02;
        G[0] = c00 / J; G[1] = c01 / J; G[2] = c02 / J;
        G[3] = (F[2] * F[7] - F[1] * F[8]) / J; G[4] = (F[0] * F[8] - F[2] * F[6]) / J;
        G[5] = (F[1] * F[6] - F[0] * F[7]) / J;
        G[6] = (F[1] * F[5] - F[2] * F[4]) / J; G[7] = (F[2] * F[3] - F[0] * F[5]) / J;
        G[8] = (F[0] * F[4] - F[1] * F[3]) / J;
    }
    const double lnJ = log(J);
    for (int i = 0; i < D; ++i)
        for (int I = 0; I < D; ++I) {
            P[i * D + I] = mu * (F[i * D + I] - G[i * D + I]) + lam * lnJ * G[i * D + I];
            for (int k = 0; k < D; ++k)
                for (int K = 0; K < D; ++K)
                    A[(i * D + I) * D2 + k * D + K] = mu * (i == k) * (I == K)
                        + (mu - lam * lnJ) * G[i * D + K] * G[k * D + I]
                        + lam * G[i * D + I] * G[k * D + K];
        }
}
"""


# ---------------------------------------- helper -------------------------------------
def gyroid(nvoxels):
    axes = [(np.arange(n) + 0.5) * length / n for n, length in zip(nvoxels, LENGTHS)]
    x, y, z = np.meshgrid(*axes, *([[0.25]] if D == 2 else []), indexing="ij")
    k = 2.0 * np.pi * CELLS / LENGTHS[0]
    level_set = (
        np.sin(k * x) * np.cos(k * y)
        + np.sin(k * y) * np.cos(k * z)
        + np.sin(k * z) * np.cos(k * x)
    )
    plates = (x < PLATE * LENGTHS[0]) | (x > (1.0 - PLATE) * LENGTHS[0])
    solid = ((np.abs(level_set) < THICKNESS) | plates).reshape(nvoxels)
    labels, _ = scipy.ndimage.label(solid)
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def face_dofs(basis, face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


def supports(basis):  # left face clamped, right face moved along x
    left = [face_dofs(basis, 0, i) for i in range(D)]
    return np.unique(np.concatenate(left + [face_dofs(basis, 1, 0)]))


# ---------------------------------------- setup --------------------------------------
nvoxels = [int(RESOLUTION * length) for length in LENGTHS]
solid = gyroid(nvoxels)
params = np.stack(
    [np.where(solid, E_SOLID, ALPHA * E_SOLID), np.full(nvoxels, NU), solid], axis=-1
)

tic = time.time()
solver = VoxelMultigrid(None, nvoxels, LENGTHS, DEGREE, SUB_VOXELS, D, supports)
cell = [length / n * SUB_VOXELS for length, n in zip(LENGTHS, nvoxels)]
tables = QuadratureTables(mlhp.makeHpTensorSpace, cell, DEGREE, SUB_VOXELS)
problem = PointwiseProblem(solver, tables, PointMaterial(MATERIAL, D, 3), params)
right = cp.asarray(face_dofs(solver.basis, 1, 0))
print(
    f"dofs {solver.ndof}, solid fraction {solid.mean():.2f}, setup {time.time() - tic:.2f} s"
)

# ------------------------------------- simulation ------------------------------------
u = cp.zeros(solver.ndof)
reactions = []
t_assembly = t_solve = 0.0
cg_iterations = []

tic = time.time()
for step in range(1, STEPS + 1):
    target = -COMPRESSION * LENGTHS[0] * step / STEPS
    norms, cg_step = [], []
    for newton in range(MAX_NEWTON):
        t0 = time.time()
        forces, K_e = problem.evaluate(u)
        solver.update_elements(K_e, hierarchy=False)
        jump = cp.zeros(solver.ndof)
        jump[right] = (
            target - u[right]
        )  # Dirichlet increment, nonzero in the first iteration
        residual = forces + solver.apply_full(jump)
        norms.append(float(cp.linalg.norm(residual * solver.mask)))
        cp.cuda.Device().synchronize()
        t_assembly += time.time() - t0
        if newton > 0 and norms[-1] <= NEWTON_TOL * norms[0]:
            break
        t0 = time.time()
        correction, it = solver.solve(-residual, rtol=CG_TOL)
        t_solve += time.time() - t0
        cg_iterations.append(it)
        cg_step.append(it)
        u += jump + correction
    reactions.append(float(forces[right].sum()))
    print(
        f"step {step} u {target:+.3f} reaction {reactions[-1]:+.4e} newton {newton} "
        f"residuals "
        + " ".join(f"{r / norms[0]:.1e}" for r in norms[1:])
        + f" cg {cg_step}"
    )
    if not np.isfinite(reactions[-1]):
        break
toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s (assembly {t_assembly:.2f} s, solve {t_solve:.2f} s), "
    f"cg iterations mean {np.mean(cg_iterations):.1f} max {max(cg_iterations)}"
)

# ----------------------------------- postprocessing ----------------------------------
# deformed voxel centers of the solid, coloured by displacement magnitude
fine = solver.levels[0]
u_e = u[fine.efts].reshape(-1, D, fine.n // D).mean(axis=2).get()
nel = fine.nel
centers = np.stack(
    np.meshgrid(
        *[(np.arange(n) + 0.5) * length / n for n, length in zip(nel, LENGTHS)],
        indexing="ij",
    ),
    -1,
).reshape(-1, D)
keep = solid.ravel()
if D == 3:
    keep &= np.abs(centers[:, 2] - 0.5 * LENGTHS[2]) < 0.5 * LENGTHS[2] / nel[2] + 1e-9
deformed = centers[keep] + u_e[keep]
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.scatter(
    *deformed[:, :2].T,
    c=np.linalg.norm(u_e[keep], axis=1),
    s=2,
    cmap="turbo",
    marker="s",
)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
