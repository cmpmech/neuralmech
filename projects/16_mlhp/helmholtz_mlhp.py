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

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = 2

DEGREE = 3
NELEMENTS = 32
WAVENUMBER = 10.0  # k = omega / c
DAMPING = 8.0  # eta in (k^2 + i eta)

# complex manufactured solution u = u_re + i u_im on the unit square, with u = 0 on the
# boundary, so that laplacian(u) + (k^2 + i eta) u = -f for the source derived below
MX, MY = 3, 2  # real-part mode
NX, NY = 2, 3  # imaginary-part mode

u_re = f"sin({MX * np.pi} * x) * sin({MY * np.pi} * y)"
u_im = f"sin({NX * np.pi} * x) * sin({NY * np.pi} * y)"

alpha_re = (MX**2 + MY**2) * np.pi**2 - WAVENUMBER**2
alpha_im = (NX**2 + NY**2) * np.pi**2 - WAVENUMBER**2

# f_re = alpha_re u_re + eta u_im,  f_im = alpha_im u_im - eta u_re
source_re = mlhp.scalarField(D, f"{alpha_re} * {u_re} + {DAMPING} * {u_im}")
source_im = mlhp.scalarField(D, f"{alpha_im} * {u_im} - {DAMPING} * {u_re}")

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
integrand = mlhphelpers.helmholtzIntegrand(wavenumber, damping, source_re, source_im)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

# ---------------------------------------- solve --------------------------------------
# the eta-coupling makes the system non-symmetric and indefinite, so solve directly
operator = sp.csr_matrix(
    (np.asarray(matrix.data_array), np.asarray(matrix.indices_array), np.asarray(matrix.indptr_array)),
    shape=tuple(matrix.shape),
)
interior_dofs = spsolve(operator, np.asarray(vector))
all_dofs = mlhp.inflateDofs(mlhp.DoubleVector(interior_dofs), dirichlet)

# ----------------------------------- postprocessing ----------------------------------
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, [mlhp.solutionProcessor(D, all_dofs)])

data = np.array(result.data()[0])
u_re_h, u_im_h = data[0::2], data[1::2]

# verify against the manufactured solution at the sample points
tri = result.triangulation()
x, y = tri.x, tri.y
u_re_ex = np.sin(MX * np.pi * x) * np.sin(MY * np.pi * y)
u_im_ex = np.sin(NX * np.pi * x) * np.sin(NY * np.pi * y)
err = np.sqrt(np.sum((u_re_h - u_re_ex) ** 2 + (u_im_h - u_im_ex) ** 2))
ref = np.sqrt(np.sum(u_re_ex**2 + u_im_ex**2))
print(f"relative error: {err / ref:.3e}")

fig, axes = plt.subplots(1, 2)
for ax, field in zip(axes, (u_re_h, u_im_h)):
    cb = ax.tricontourf(tri, field, cmap="seismic", levels=24)
    fig.colorbar(cb, ax=ax)
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout(pad=0)

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "helmholtz_mlhp.pdf")
else:
    plt.show()
