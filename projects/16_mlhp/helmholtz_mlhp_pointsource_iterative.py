import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import mlhphelpers
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
# Iterative variant of helmholtz_mlhp_pointsource.py. The system is non-symmetric
# (the eta-coupling blocks), so CG does not apply; BiCGStab is used with a diagonal
# (Jacobi) preconditioner. Both are pure sparse mat-vec + vector ops + elementwise
# scaling, i.e. they port directly to a matrix-free GPU implementation.
#
# Convergence regime (empirical, degree-3, 64x64 grid): reliable for low-to-moderate
# wavenumbers (k <~ 30) with light damping. Strong damping HURTS here because the
# eta-coupling sits in the off-diagonal real/imaginary field blocks, which the
# diagonal preconditioner does not see; once eta dominates, BiCGStab breaks down
# (look for "<r0, v> is 0"). For k >= 50 or eta >~ 0.1 k^2 expect failure.
D = 2

DEGREE = 3
NELEMENTS = 64
WAVENUMBER = 20.0  # k = omega / c, kept in the convergent regime
DAMPING = 20.0  # eta in (k^2 + i eta), light relative to k^2 = 400

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
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0, 1, 2, 3], ifield=ifield)
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
# BiCGStab + Jacobi: both matrix-free-friendly (SpMV, AXPY, dot, 1/diag scaling)
preconditioner = mlhp.diagonalPreconditioner(matrix)
interior_dofs, residuals = mlhp.bicgstab(
    matrix, vector, M=preconditioner, rtol=1e-6, maxiter=5000, residualNorms=True
)
all_dofs = mlhp.inflateDofs(interior_dofs, dirichlet)
print(f"BiCGStab: {len(residuals)} iterations, relative residual {residuals[-1] / residuals[0]:.1e}")

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
    tri, magnitude, cmap="inferno", levels=np.linspace(0, np.percentile(magnitude, 99), 24),
    extend="max",
)
fig.colorbar(cb1, ax=axes[1])
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)  # vectorized pdf too large at this mesh density
fig.tight_layout(pad=0)

if args.book:
    fig.savefig(RGB_PDF_DIR / "helmholtz_mlhp_pointsource_iterative.pdf")
else:
    plt.show()
