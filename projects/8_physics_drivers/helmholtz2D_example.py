import mlhp

# import mlhphelpers
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

D = 2

# Setup discretization
polynomialDegree = 3
nelements = [64] * D
wavenumber = 140.0  # 80.0  # 40.0  # k = omega / c
damping = 10.0  # 10.0  # 1.0  # eta in (k^2 + i eta)
alphaFCM = 1e-8

# unit square with a circular hole punched out of it
origin, maximum = [0.0] * D, [1.0] * D
cube = mlhp.implicitCube(origin, maximum)
hole = mlhp.implicitSphere([0.5, 0.5], 0.15)
domain = mlhp.implicitSubtraction([cube, hole])

# point source: narrow Gaussian, offset so it does not fall inside the hole
center, width, amplitude = [0.25, 0.25], 0.01, 1.0
radius2 = f"((x - {center[0]})**2 + (y - {center[1]})**2)"
source = mlhp.scalarField(D, f"{amplitude} * exp(-{radius2} / (2 * {width}**2))")

# fictitious-domain mesh: keep only cells that intersect the holed domain
baseGrid = mlhp.makeGrid(nelements, [1.0] * D, origin)
mesh = mlhp.makeRefinedGrid(
    mlhp.makeFilteredGrid(baseGrid, domain=domain, nseedpoints=polynomialDegree + 2)
)
basis = mlhp.makeHpTrunkSpace(mesh, degree=polynomialDegree, nfields=2)
print(basis)

# homogeneous Neumann boundary conditions
dirichlet = mlhp.combineDirichletDofs([])

# Assemble
integrand = mlhp.helmholtzIntegrand(
    mlhp.scalarField(D, wavenumber), mlhp.scalarField(D, damping), source
)
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

# integrate over the cut domain (finite-cell quadrature around the hole)
quadrature = mlhp.spaceTreeQuadrature(
    domain, depth=polynomialDegree + 1, epsilon=alphaFCM
)
mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

# Solve directly (system is non-symmetric / indefinite)
operator = sp.csr_matrix(
    (
        np.asarray(matrix.data_array),
        np.asarray(matrix.indices_array),
        np.asarray(matrix.indptr_array),
    ),
    shape=tuple(matrix.shape),
)
interiorDofs = spsolve(operator, np.asarray(vector))
allDofs = mlhp.inflateDofs(mlhp.DoubleVector(interiorDofs), dirichlet)

# Post-processing: plot the real part
import matplotlib.pyplot as plt

postmesh = mlhp.domainCellMesh(domain, [polynomialDegree + 2] * D)
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, result, [mlhp.solutionProcessor(D, allDofs)])

data = np.array(result.data()[0])
u_re, u_im = data[0::2], data[1::2]
phase = np.arctan2(u_im, u_re)  # phase shift in (-pi, pi]
tri = result.triangulation()

limit_re = np.max(np.abs(u_re))
limit_im = np.max(np.abs(u_im))
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.tricontourf(tri, u_re, cmap="seismic", levels=np.linspace(-limit_re, limit_re, 64))
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig("../../results/helmholtz2D_amp.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.tricontourf(tri, u_im, cmap="seismic", levels=np.linspace(-limit_im, limit_im, 64))
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig("../../results/helmholtz2D_phase.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
