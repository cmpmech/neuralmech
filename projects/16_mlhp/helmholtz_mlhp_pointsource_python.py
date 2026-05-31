import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse as sp

try:
    from pypardiso import spsolve  # MKL PARDISO: multithreaded, ~20x faster
except ImportError:
    from scipy.sparse.linalg import spsolve

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
# Pure-python variant of helmholtz_mlhp_pointsource.py: no custom C++ integrand. The
# stock poissonIntegrand and l2DomainIntegrand give the scalar stiffness K and mass M
# on a single-field basis; the damped complex system is then assembled as the real
# 2x2 block matrix [[K-k^2 M, eta M], [-eta M, K-k^2 M]] with scipy and solved directly.
D = 2

DEGREE = 3
NELEMENTS = 64
WAVENUMBER = 30.0  # k = omega / c, wavelength 2 pi / k
DAMPING = 10.0  # eta in (k^2 + i eta)

# point excitation approximated by a narrow Gaussian centered in the domain
CENTER = 0.5
WIDTH = 0.01
AMPLITUDE = 1.0

radius2 = f"((x - {CENTER})**2 + (y - {CENTER})**2)"
source = mlhp.scalarField(D, f"{AMPLITUDE} * exp(-{radius2} / (2 * {WIDTH}**2))")

# ------------------------------------- discretization --------------------------------
mesh = mlhp.makeRefinedGrid(ncells=[NELEMENTS] * D, lengths=[1.0] * D)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE)  # single scalar field
print(basis)

dirichlet = mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0, 1, 2, 3])

# --------------------------------------- assembly ------------------------------------
def to_csr(matrix):
    return sp.csr_matrix(
        (np.asarray(matrix.data_array), np.asarray(matrix.indices_array), np.asarray(matrix.indptr_array)),
        shape=tuple(matrix.shape),
    )

# stiffness K and load vector F from the Poisson integrand (unit conductivity)
k_matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
f_vector = mlhp.allocateRhsVector(k_matrix)
mlhp.integrateOnDomain(
    basis, mlhp.poissonIntegrand(mlhp.scalarField(D, 1.0), source),
    [k_matrix, f_vector], dirichletDofs=dirichlet,
)

# mass M from the L2 integrand (unit mass, no source)
m_matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
mlhp.integrateOnDomain(
    basis, mlhp.l2DomainIntegrand(mass=mlhp.scalarField(D, 1.0)),
    [m_matrix], dirichletDofs=dirichlet,
)

K, M, F = to_csr(k_matrix), to_csr(m_matrix), np.asarray(f_vector)

# real 2x2 block system for u = u_re + i u_im
H = K - WAVENUMBER**2 * M
operator = sp.bmat([[H, DAMPING * M], [-DAMPING * M, H]], format="csr")
rhs = np.concatenate([F, np.zeros_like(F)])

# ---------------------------------------- solve --------------------------------------
x = spsolve(operator, rhs)
n = K.shape[0]
all_re = mlhp.inflateDofs(mlhp.DoubleVector(x[:n]), dirichlet)
all_im = mlhp.inflateDofs(mlhp.DoubleVector(x[n:]), dirichlet)

# ----------------------------------- postprocessing ----------------------------------
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)
result = mlhp.DataAccumulator()
mlhp.basisOutput(
    basis, postmesh, result,
    [mlhp.solutionProcessor(D, all_re), mlhp.solutionProcessor(D, all_im)],
)

u_re = np.array(result.data()[0])
u_im = np.array(result.data()[1])
tri = result.triangulation()

# real part (oscillating field) and magnitude (decaying envelope)
magnitude = np.sqrt(u_re**2 + u_im**2)
limit = np.percentile(np.abs(u_re), 99)

fig, axes = plt.subplots(1, 2)
cb0 = axes[0].tricontourf(
    tri, u_re, cmap="seismic", levels=np.linspace(-limit, limit, 24), extend="both"
)
fig.colorbar(cb0, ax=axes[0])
cb1 = axes[1].tricontourf(
    tri, magnitude, cmap="inferno", levels=np.linspace(0, np.percentile(magnitude, 99), 24),
    extend="max",
)
fig.colorbar(cb1, ax=axes[1])
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout(pad=0)

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "helmholtz_mlhp_pointsource_python.pdf")
else:
    plt.show()
