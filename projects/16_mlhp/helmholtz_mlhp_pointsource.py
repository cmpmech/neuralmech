import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import mlhphelpers
import numpy as np
import scipy.sparse as sp

try:
    from pypardiso import spsolve  # MKL PARDISO: multithreaded, ~20x faster
except ImportError:
    from scipy.sparse.linalg import spsolve

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = 2

DEGREE = 3
NELEMENTS = 64
WAVENUMBER = 80.0  # k = omega / c, wavelength 2 pi / k
DAMPING = 1.0  # 120.0  # eta in (k^2 + i eta); attenuation length ~ 2 k / eta

# point excitation approximated by a narrow Gaussian centered in the domain
CENTER = 0.5
WIDTH = 0.01
AMPLITUDE = 1.0

radius2 = f"((x - {CENTER})**2 + (y - {CENTER})**2)"
source = mlhp.scalarField(D, f"{AMPLITUDE} * exp(-{radius2} / (2 * {WIDTH}**2))")

# ------------------------------------- discretization --------------------------------
mesh = mlhp.makeRefinedGrid(ncells=[NELEMENTS] * D, lengths=[1.0] * D)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=2)
print(basis)

# both real and imaginary fields vanish on all four sides
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [0, 1, 2, 3], ifield=ifield
    )
    for ifield in (0, 1)
]
dirichlet = mlhp.combineDirichletDofs(bc_list)

# --------------------------------------- assembly ------------------------------------
wavenumber = mlhp.scalarField(D, WAVENUMBER)
damping = mlhp.scalarField(D, DAMPING)
# real-valued excitation -> imaginary source defaults to zero
integrand = mlhphelpers.helmholtzIntegrand(wavenumber, damping, source)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

# ---------------------------------------- solve --------------------------------------
# the eta-coupling makes the system non-symmetric and indefinite, so solve directly
operator = sp.csr_matrix(
    (
        np.asarray(matrix.data_array),
        np.asarray(matrix.indices_array),
        np.asarray(matrix.indptr_array),
    ),
    shape=tuple(matrix.shape),
)
interior_dofs = spsolve(operator, np.asarray(vector))
all_dofs = mlhp.inflateDofs(mlhp.DoubleVector(interior_dofs), dirichlet)

# ----------------------------------- postprocessing ----------------------------------
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, [mlhp.solutionProcessor(D, all_dofs)])

data = np.array(result.data()[0])
u_re, u_im = data[0::2], data[1::2]
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
    tri,
    magnitude,
    cmap="inferno",
    levels=np.linspace(0, np.percentile(magnitude, 99), 24),
    extend="max",
)
fig.colorbar(cb1, ax=axes[1])
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)  # vectorized pdf too large at this mesh density
fig.tight_layout(pad=0)

if args.book:
    fig.savefig(RGB_PDF_DIR / "helmholtz_mlhp_pointsource.pdf")
else:
    plt.show()
