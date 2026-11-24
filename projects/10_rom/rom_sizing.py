import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse

from postprocessing import save_csv
from solvers.optimization import StructuredFEM

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# geometry
LENGTHS = [2.0, 1.0]
REGIONS = (8, 4)  # stiffness regions per direction, the design parameters

# discretization
RESOLUTION = 160  # elements across the height
NU = 0.3

# design space
STIFFNESS = (0.1, 1.0)  # bounds of the Young's modulus of every region
BUDGET = 0.5  # mean Young's modulus

# reduced model
SNAPSHOTS = 64  # random designs for the basis
TESTS = 100  # random designs for the error
RANKS = np.arange(1, 41)
TOL = 1e-4  # error of the reduced model at its optimum that ends the enrichment

# optimization (optimality criterion)
MOVE = 0.2
ITERS = 100

# --------------------------------------- setup ---------------------------------------
rng = np.random.default_rng(0)
nely = RESOLUTION
nelx = int(round(LENGTHS[0] / LENGTHS[1])) * nely
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx, nely], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=1, nfields=2)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

mesh_local = mlhp.makeRefinedGrid(
    mlhp.makeGrid(ncells=[1, 1], lengths=[LENGTHS[0] / nelx, LENGTHS[1] / nely])
)
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=1, nfields=2)
material = mlhp.planeStressMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
K_locals = mlhp.integratePartitionMatrices(
    basis_local,
    integrand,
    mlhp.gridQuadrature(nsubcells=[1, 1]),
    mlhp.absoluteQuadratureOrder([2, 2]),
)


def face_dofs(face, ifield):  # faces: 0=left, 1=right, 2=bottom, 3=top
    bc = mlhp.integrateDirichletDofs(mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield)
    return np.array(mlhp.combineDirichletDofs([bc])[0])


fixed = np.unique(np.concatenate([face_dofs(0, 0), face_dofs(0, 1)]))
free = np.setdiff1d(np.arange(ndof), fixed)
load = face_dofs(1, 1)
force = np.zeros(ndof)
force[load] = -1 / len(load)  # shear load on the right edge
f = force[free]

fem = StructuredFEM(efts, free, ndof, K_locals, (nelx, nely))
ix, iy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
region = ix // (nelx // REGIONS[0]) * REGIONS[1] + iy // (nely // REGIONS[1])
P = REGIONS[0] * REGIONS[1]


def full_matrix(data):  # pardiso stores the upper triangle only
    upper = fem.matrix(data)
    return upper + upper.T - scipy.sparse.diags(upper.diagonal())


K_parts = [full_matrix(fem.data((region == k).astype(float), K_locals)) for k in range(P)]


def to_grid(mu):
    return mu[region]


def solve_full(mu):
    return fem.solve(to_grid(mu), f)


def random_designs(n):
    return rng.uniform(*STIFFNESS, size=(n, P))


# ------------------------------ random snapshots & basis -----------------------------
X = np.array([solve_full(mu) for mu in random_designs(SNAPSHOTS)]).T
Phi_all, S, _ = np.linalg.svd(X, full_matrices=False)


def reduced_operators(Phi):  # affine in mu, so the reduced parts are computed once
    return np.array([Phi.T @ (K @ Phi) for K in K_parts]), Phi.T @ f


def solve_reduced(mu, K_r, f_r):
    return np.linalg.solve(np.tensordot(mu, K_r, 1), f_r)


tests = random_designs(TESTS)
U_test = np.array([solve_full(mu) for mu in tests])
errors_random = []
for rank in RANKS:
    Phi = Phi_all[:, :rank]
    K_r, f_r = reduced_operators(Phi)
    U_rom = np.array([Phi @ solve_reduced(mu, K_r, f_r) for mu in tests])
    errors_random.append(np.linalg.norm(U_rom - U_test) / np.linalg.norm(U_test))


# ------------------------------------ optimization -----------------------------------
def optimize(solve, energies, mu):
    for _ in range(ITERS):
        dc = -energies(solve(mu))  # self-adjoint: dc/dmu_k = -u^T K_k u
        lo = np.maximum(STIFFNESS[0], mu - MOVE)
        hi = np.minimum(STIFFNESS[1], mu + MOVE)
        l1, l2 = 1e-9, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            mu_new = np.clip(mu * np.sqrt(-dc / lmid), lo, hi)
            l1, l2 = (lmid, l2) if mu_new.mean() > BUDGET else (l1, lmid)
        mu = mu_new
    return mu


tic = time.perf_counter()
mu_full = optimize(
    solve_full, lambda u: np.array([u @ (K @ u) for K in K_parts]), np.full(P, BUDGET)
)
time_full = time.perf_counter() - tic
compliance_full = f @ solve_full(mu_full)

# reduced model from random snapshots, checked by a full solve at its optimum
Phi = Phi_all[:, :20]
K_r, f_r = reduced_operators(Phi)
mu_random = optimize(
    lambda m: solve_reduced(m, K_r, f_r),
    lambda c: np.einsum("a,kab,b->k", c, K_r, c),
    np.full(P, BUDGET),
)
u = solve_full(mu_random)
error_random = np.linalg.norm(Phi @ solve_reduced(mu_random, K_r, f_r) - u) / np.linalg.norm(u)
gap_random = f @ u / compliance_full - 1

# greedy enrichment: optimize on the reduced model, add the full solution at its optimum
tic = time.perf_counter()
mu = np.full(P, BUDGET)
snapshots = [solve_full(mu)]
errors_greedy, gaps = [], []
while True:
    Phi = np.linalg.qr(np.array(snapshots).T)[0]
    K_r, f_r = reduced_operators(Phi)
    mu = optimize(
        lambda m: solve_reduced(m, K_r, f_r),
        lambda c: np.einsum("a,kab,b->k", c, K_r, c),
        mu,
    )
    u = solve_full(mu)
    errors_greedy.append(np.linalg.norm(Phi @ solve_reduced(mu, K_r, f_r) - u) / np.linalg.norm(u))
    gaps.append(f @ u / compliance_full - 1)
    if errors_greedy[-1] < TOL:
        break
    snapshots.append(u)
time_greedy = time.perf_counter() - tic
mu_rom = mu

print(f"{free.size} dofs, {P} parameters, random snapshots: error {errors_random[19]:.1e} at R = 20")
print(f"random basis (R = 20): error {error_random:.1e} at its optimum, compliance gap {gap_random:.1e}")
print(f"full optimization: {ITERS} full solves in {time_full:.2f} s, compliance {compliance_full:.4f}")
print(f"greedy: {len(snapshots)} full solves (R = {Phi.shape[1]}) in {time_greedy:.2f} s")
print(f"compliance gap {gaps[-1]:.1e}, design difference {np.abs(mu_rom - mu_full).max():.1e}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(RANKS, errors_random, "k")
    axes[0].semilogy(np.arange(1, len(errors_greedy) + 1), errors_greedy, "r.-")
    axes[1].imshow(to_grid(mu_rom).T, origin="lower", cmap="cividis", vmin=0, vmax=1)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "rom_sizing_random.csv", R=RANKS, error=errors_random)
    save_csv(
        CSV_DIR / "rom_sizing_greedy.csv",
        R=np.arange(1, len(errors_greedy) + 1),
        error=errors_greedy,
        gap=np.abs(gaps),
    )
    fig, ax = plt.subplots(figsize=(2 * LENGTHS[0], 2 * LENGTHS[1]))
    ax.imshow(to_grid(mu_rom).T, origin="lower", cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RGB_PDF_DIR / "rom_sizing_design.pdf", dpi=300)
