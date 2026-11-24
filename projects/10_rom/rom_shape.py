import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.sparse as sp

from postprocessing import save_csv
from solvers.mklwrapper import pardisoFactorize

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# geometry: a frame of 3 x 3 blocks around a hole
LENGTHS = (3.0, 1.0)
SOLID = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # [column, row]
MIN_SIZE = 0.1  # smallest column width and row height

# discretization
ELEMENTS = 64  # elements per block and direction, whatever its size
NU = 0.3
VOID = 1e-6  # stiffness of the holes

# reduced model
SNAPSHOTS = 100  # random shapes for the basis
TESTS = 50  # random shapes for the error
RANKS = np.arange(1, 61)
R = 40  # rank used in the optimization
TOL = 1e-2  # error at an optimum above which its full solution enlarges the basis

# optimization: compliance at several prescribed areas, a Pareto front
AREAS = np.linspace(1.6, 2.6, 11)

# --------------------------------------- setup ---------------------------------------
rng = np.random.default_rng(0)
cols, rows = SOLID.shape
Nx, Ny = cols * ELEMENTS, rows * ELEMENTS


def element_matrices():
    """unit-modulus Q4 matrices K_xixi, K_etaeta, K_xieta of a b x a rectangle: (b/a) K_xixi + (a/b) K_etaeta + K_xieta"""
    C = np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]]) / (1 - NU**2)
    xi, eta = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    g = 1 / np.sqrt(3)
    Ks = [np.zeros((8, 8)) for _ in range(3)]
    for x in (-g, g):
        for y in (-g, g):
            dxi, deta = xi * (1 + eta * y) / 4, eta * (1 + xi * x) / 4
            Bxi, Beta = np.zeros((3, 8)), np.zeros((3, 8))
            Bxi[0, 0::2], Bxi[2, 1::2] = dxi, dxi
            Beta[1, 1::2], Beta[2, 0::2] = deta, deta
            Ks[0] += Bxi.T @ C @ Bxi * 4
            Ks[1] += Beta.T @ C @ Beta * 4
            Ks[2] += (Bxi.T @ C @ Beta + Beta.T @ C @ Bxi) * 4
    return Ks


i, j = np.divmod(np.arange(Nx * Ny), Ny)
n0 = i * (Ny + 1) + j
edofs = (2 * np.stack([n0, n0 + Ny + 1, n0 + Ny + 2, n0 + 1], axis=1)[:, :, None] + np.arange(2)).reshape(-1, 8)
block = (i // ELEMENTS) * rows + j // ELEMENTS
ndof = 2 * (Nx + 1) * (Ny + 1)
free = np.arange(2 * (Ny + 1), ndof)  # the left edge is clamped

# per block, the three unit matrices on the free dofs
K_terms = []
for k in range(cols * rows):
    e = edofs[block == k]
    r_idx, c_idx = np.repeat(e, 8, axis=1).ravel(), np.tile(e, 8).ravel()
    for Ke in element_matrices():
        K = sp.csr_matrix((np.tile(Ke.ravel(), len(e)), (r_idx, c_idx)), shape=(ndof, ndof))
        K_terms.append(K[free][:, free])

# shear load on the right edge, per unit length of each row: f = sum_j h_j f_j
f_terms = []
for jr in range(rows):
    f = np.zeros(ndof)
    nodes = Nx * (Ny + 1) + jr * ELEMENTS + np.arange(ELEMENTS + 1)
    w = np.full(ELEMENTS + 1, 1.0 / ELEMENTS)
    w[[0, -1]] /= 2
    f[2 * nodes + 1] = -w / LENGTHS[1]
    f_terms.append(f[free])
f_terms = np.array(f_terms)


def coefficients(mu):  # mu = (column widths, row heights): theta of every term, the load
    widths, heights = mu[:cols], mu[cols:]
    a = np.repeat(widths, rows) / ELEMENTS
    b = np.tile(heights, cols) / ELEMENTS
    E = np.where(SOLID.ravel() == 1, 1.0, VOID)
    theta = np.stack([E * b / a, E * a / b, E], axis=1).ravel()
    return theta, heights


def dcoefficients(mu):  # d theta / d mu, (len(mu), terms), and d f-weights / d mu
    widths, heights = mu[:cols], mu[cols:]
    a = np.repeat(widths, rows) / ELEMENTS
    b = np.tile(heights, cols) / ELEMENTS
    E = np.where(SOLID.ravel() == 1, 1.0, VOID)
    dtheta = np.zeros((len(mu), 3 * cols * rows))
    for k in range(cols * rows):
        c, r = divmod(k, rows)
        dtheta[c, 3 * k] = -E[k] * b[k] / a[k] / widths[c]
        dtheta[c, 3 * k + 1] = E[k] * a[k] / b[k] / widths[c]
        dtheta[cols + r, 3 * k] = E[k] * b[k] / a[k] / heights[r]
        dtheta[cols + r, 3 * k + 1] = -E[k] * a[k] / b[k] / heights[r]
    dweights = np.zeros((len(mu), rows))
    dweights[cols:] = np.eye(rows)
    return dtheta, dweights


def solve_full(mu):
    theta, weights = coefficients(mu)
    K = sum(t * Kt for t, Kt in zip(theta, K_terms))
    f = weights @ f_terms
    return pardisoFactorize(sp.triu(K, format="csr"), symmetric=True, symmetricHalf=True)(f), f


def area(mu):
    return mu[:cols] @ SOLID @ mu[cols:]


def random_shapes(n):
    widths = rng.dirichlet(np.full(cols, 2.0), n) * (LENGTHS[0] - cols * MIN_SIZE) + MIN_SIZE
    heights = rng.dirichlet(np.full(rows, 2.0), n) * (LENGTHS[1] - rows * MIN_SIZE) + MIN_SIZE
    return np.hstack([widths, heights])


# ------------------------------------ reduced model ----------------------------------
tic = time.perf_counter()
X = np.array([solve_full(mu)[0] for mu in random_shapes(SNAPSHOTS)]).T
Phi_all, S, _ = np.linalg.svd(X, full_matrices=False)
time_snapshots = time.perf_counter() - tic


def reduced_operators(Phi):  # the affine terms projected once
    return np.array([Phi.T @ (K @ Phi) for K in K_terms]), f_terms @ Phi


def solve_reduced(mu, K_r, f_r):
    theta, weights = coefficients(mu)
    f = weights @ f_r
    return np.linalg.solve(np.tensordot(theta, K_r, 1), f), f


tests = random_shapes(TESTS)
U_test = np.array([solve_full(mu)[0] for mu in tests])
errors = []
for rank in RANKS:
    Phi = Phi_all[:, :rank]
    K_r, f_r = reduced_operators(Phi)
    U_rom = np.array([Phi @ solve_reduced(mu, K_r, f_r)[0] for mu in tests])
    errors.append(np.linalg.norm(U_rom - U_test) / np.linalg.norm(U_test))
tic = time.perf_counter()
Phi = Phi_all[:, :R]
K_r, f_r = reduced_operators(Phi)
time_projection = time.perf_counter() - tic
print(f"{free.size} dofs, {len(K_terms)} affine terms, error at R = {R}: {errors[R - 1]:.1e}")


# ------------------------------------ optimization -----------------------------------
def compliance_full(mu):
    u, f = solve_full(mu)
    dtheta, dweights = dcoefficients(mu)
    energies = np.array([u @ (K @ u) for K in K_terms])
    return f @ u, 2 * dweights @ (f_terms @ u) - dtheta @ energies


def compliance_reduced(mu):
    c, f = solve_reduced(mu, K_r, f_r)
    dtheta, dweights = dcoefficients(mu)
    energies = np.einsum("a,kab,b->k", c, K_r, c)
    return f @ c, 2 * dweights @ (f_r @ c) - dtheta @ energies


def optimize(compliance, target, start=None):
    constraints = [
        {"type": "eq", "fun": lambda mu: mu[:cols].sum() - LENGTHS[0], "jac": lambda mu: np.r_[np.ones(cols), np.zeros(rows)]},
        {"type": "eq", "fun": lambda mu: mu[cols:].sum() - LENGTHS[1], "jac": lambda mu: np.r_[np.zeros(cols), np.ones(rows)]},
        {"type": "eq", "fun": lambda mu: area(mu) - target, "jac": lambda mu: np.r_[SOLID @ mu[cols:], mu[:cols] @ SOLID]},
    ]
    start = feasible_start(target) if start is None else start
    result = scipy.optimize.minimize(
        compliance, start, jac=True, method="SLSQP", bounds=[(MIN_SIZE, None)] * (cols + rows),
        constraints=constraints, options={"maxiter": 200, "ftol": 1e-10},
    )
    return result.x, result.fun, result.nfev


def feasible_start(target):  # equal columns, the hole height set by the area
    widths = np.full(cols, LENGTHS[0] / cols)
    hole = (LENGTHS[0] * LENGTHS[1] - target) / (widths @ (1 - SOLID[:, 1]))
    return np.r_[widths, (LENGTHS[1] - hole) / 2, hole, (LENGTHS[1] - hole) / 2]


tic = time.perf_counter()
results_full = [optimize(compliance_full, target) for target in AREAS]
time_full = time.perf_counter() - tic
designs_full = np.array([r[0] for r in results_full])
compliances_full = np.array([r[1] for r in results_full])

# reduced optimization, every optimum checked by one full solve
tic = time.perf_counter()
designs_rom, compliances_check, checks = [], [], 0
for target in AREAS:
    mu = None
    while True:
        mu = optimize(compliance_reduced, target, mu)[0]
        u, f = solve_full(mu)
        checks += 1
        c = solve_reduced(mu, K_r, f_r)[0]
        if np.linalg.norm(Phi @ c - u) < TOL * np.linalg.norm(u):
            break
        Phi = np.linalg.qr(np.column_stack([Phi, u]))[0]  # enlarge the basis
        K_r, f_r = reduced_operators(Phi)
    designs_rom.append(mu)
    compliances_check.append(f @ u)
time_rom = time.perf_counter() - tic
designs_rom, compliances_check = np.array(designs_rom), np.array(compliances_check)
print(f"full: {sum(r[2] for r in results_full)} solves in {time_full:.1f} s")
print(f"reduced: {SNAPSHOTS} snapshots in {time_snapshots:.1f} s, {checks} checks, R = {Phi.shape[1]}, optimization {time_rom:.1f} s")
print(f"compliance gap {np.abs(compliances_check / compliances_full - 1).max():.1e}")

# ----------------------------------- postprocessing ----------------------------------
mid = len(AREAS) // 2


def draw(ax, mu):
    x = np.r_[0, np.cumsum(mu[:cols])]
    y = np.r_[0, np.cumsum(mu[cols:])]
    for c in range(cols):
        for r in range(rows):
            if SOLID[c, r]:
                ax.fill_between([x[c], x[c + 1]], y[r], y[r + 1], color="k", lw=0)
    ax.set_aspect("equal")
    ax.axis("off")


if not args.book:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(RANKS, errors, "k")
    axes[1].plot(AREAS, compliances_full, "k")
    axes[1].plot(AREAS, compliances_check, "r--")
    draw(axes[2], designs_rom[mid])
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "rom_shape_error.csv", R=RANKS, error=errors, singular=S[: len(RANKS)] / S[0])
    save_csv(CSV_DIR / "rom_shape_front.csv", area=AREAS, full=compliances_full, rom=compliances_check)
    for k in (0, mid, len(AREAS) - 1):
        fig, ax = plt.subplots(figsize=(2 * LENGTHS[0], 2 * LENGTHS[1]))
        draw(ax, designs_rom[k])
        ax.set_xlim(0, LENGTHS[0])
        ax.set_ylim(0, LENGTHS[1])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"rom_shape_design{k}.pdf")
        plt.close(fig)
