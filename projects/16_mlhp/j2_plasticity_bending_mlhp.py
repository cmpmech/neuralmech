import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import mlhphelpers
import numpy as np
import scipy.sparse as sp

try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
# Cantilever bent into small-strain J2 plasticity: the fixed end is clamped, the tip is
# pushed down in displacement-controlled load steps. Bending puts the top/bottom fibers
# near the support past yield while the neutral plane stays elastic, giving a non-uniform
# effective-plastic-strain field that is plotted as a contour over the mid-plane.
D = 3

DEGREE = 1
LENGTHS = [4.0, 1.0, 1.0]
NELEMENTS = [24, 1, 6]

E = 210.0e3  # Young's modulus
NU = 0.3
YIELD = 250.0  # initial yield stress
HARDENING = E / 50.0  # linear isotropic hardening modulus H
HARDENING_RATIO = 0.0  # 0 = isotropic, 1 = kinematic

TIP_DEFLECTION = 0.12  # downward displacement of the tip
NSTEPS = 24
NEWTON_TOL = 1e-8
PENALTY = 1e8 * E
QUAD_ORDER = DEGREE + 1

# ------------------------------------- discretization --------------------------------
mesh = mlhp.makeRefinedGrid(ncells=NELEMENTS, lengths=LENGTHS)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
print(basis)

domain = mlhp.implicitSphere([0.5 * length for length in LENGTHS], 100.0)  # whole beam inside
kinematics = mlhp.smallStrainKinematics(D)
history = mlhphelpers.initialJ2History(mesh)

force = mlhp.vectorField(D, [0.0] * D)
ndof = basis.ndof()

# clamp the x=0 face (face 0, all components) and push the x=L tip (face 1) down in z
def boundary_conditions(deflection):
    return [(0, 0, 0.0), (0, 1, 0.0), (0, 2, 0.0), (1, 2, deflection)]

def to_csr(matrix):
    return sp.csr_matrix(
        (np.asarray(matrix.data_array), np.asarray(matrix.indices_array), np.asarray(matrix.indptr_array)),
        shape=tuple(matrix.shape),
    )

# ----------------------------------- load stepping -----------------------------------
dofs0 = mlhp.DoubleVector([0.0] * ndof)

for istep in range(1, NSTEPS + 1):
    deflection = -TIP_DEFLECTION * istep / NSTEPS
    dofs1 = mlhp.DoubleVector(list(dofs0))

    for inewton in range(25):
        matrix = mlhp.allocateSparseMatrix(basis)
        vector = mlhp.allocateRhsVector(matrix)

        increment = mlhp.DoubleVector(list(np.asarray(dofs1) - np.asarray(dofs0)))
        material = mlhphelpers.j2Material(
            mlhp.scalarField(D, E), mlhp.scalarField(D, NU), YIELD,
            hardeningModulus=HARDENING, hardeningRatio=HARDENING_RATIO,
            history=history, domain=domain,
        )
        integrand = mlhp.staticDomainIntegrand(kinematics, material, increment, force)
        mlhp.integrateOnDomain(basis, integrand, [matrix, vector])

        # penalty boundary conditions (nonlinear residual form)
        for face, ifield, target in boundary_conditions(deflection):
            bc = mlhp.l2BoundaryIntegrand(
                mlhp.scalarField(D, PENALTY), mlhp.scalarField(D, PENALTY * target),
                dofs=dofs1, ifield=ifield,
            )
            mlhp.integrateOnSurface(basis, bc, [matrix, vector], mlhp.quadratureOnMeshFaces(mesh, [face]))

        residual = np.linalg.norm(np.asarray(vector))
        if inewton == 0:
            residual0 = residual
        if residual <= max(NEWTON_TOL * residual0, 1e-12):
            break

        delta = spsolve(to_csr(matrix), np.asarray(vector))
        dofs1 = mlhp.DoubleVector(list(np.asarray(dofs1) + delta))

    # freeze the converged plastic state into the history
    history = mlhphelpers.updateJ2History(
        history, basis, dofs0, dofs1, kinematics, domain,
        mlhp.scalarField(D, E), mlhp.scalarField(D, NU), YIELD,
        HARDENING, HARDENING_RATIO, QUAD_ORDER,
    )
    dofs0 = dofs1
    print(f"step {istep:2d}: tip deflection {deflection:+.4f}, newton {inewton}")

# ----------------------------------- postprocessing ----------------------------------
# effective plastic strain (history component 12) on the y mid-plane
nx, nz = 200, 60
xs = np.linspace(0.0, LENGTHS[0], nx)
zs = np.linspace(0.0, LENGTHS[2], nz)
gx, gz = np.meshgrid(xs, zs)
ymid = 0.5 * LENGTHS[1]
points = [[x, ymid, z] for x, z in zip(gx.ravel(), gz.ravel())]

data = np.array(mlhphelpers.evaluateJ2History(history, points)).reshape(-1, 13)
plastic_strain = data[:, 12].reshape(gz.shape)
print(f"max effective plastic strain: {plastic_strain.max():.4e}")

fig, ax = plt.subplots(figsize=(8, 2.4))
cb = ax.contourf(gx, gz, plastic_strain, levels=24, cmap="magma")
fig.colorbar(cb, ax=ax)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0)

if args.book:
    fig.savefig(RGB_PDF_DIR / "j2_plasticity_bending_mlhp.pdf")
else:
    plt.show()
