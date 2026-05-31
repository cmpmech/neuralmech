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

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
# Uniaxial tension of a cube driven into small-strain J2 plasticity. Symmetry BCs make
# the stress uniform; the displacement of the top face is ramped past yield and the
# resulting axial stress (read back from the return-mapped history) is plotted against
# strain to validate the material against the analytical elastic/hardening response.
# See j2_plasticity_bending_mlhp.py for a non-uniform (bending) plastic-strain field.
D = 3

DEGREE = 1
NELEMENTS = 4
LENGTH = 1.0

E = 210.0e3  # Young's modulus
NU = 0.3
YIELD = 250.0  # initial yield stress
HARDENING = E / 50.0  # linear isotropic hardening modulus H
HARDENING_RATIO = 0.0  # 0 = isotropic, 1 = kinematic

STRAIN_MAX = 0.005  # ~4x the yield strain
NSTEPS = 20
NEWTON_TOL = 1e-8
PENALTY = 1e8 * E
QUAD_ORDER = DEGREE + 1

# ------------------------------------- discretization --------------------------------
mesh = mlhp.makeRefinedGrid(ncells=[NELEMENTS] * D, lengths=[LENGTH] * D)
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
print(basis)

domain = mlhp.implicitSphere([0.5 * LENGTH] * D, 100.0)  # whole cube is inside
kinematics = mlhp.smallStrainKinematics(D)
history = mlhphelpers.initialJ2History(mesh)

force = mlhp.vectorField(D, [0.0] * D)
ndof = basis.ndof()

# symmetry planes (u_n = 0) plus the driven top face (u_z = applied); (face, ifield, target)
def boundary_conditions(applied):
    return [
        (0, 0, 0.0),  # x-min: u_x = 0
        (2, 1, 0.0),  # y-min: u_y = 0
        (4, 2, 0.0),  # z-min: u_z = 0
        (5, 2, applied),  # z-max: u_z = applied
    ]

def to_csr(matrix):
    return sp.csr_matrix(
        (np.asarray(matrix.data_array), np.asarray(matrix.indices_array), np.asarray(matrix.indptr_array)),
        shape=tuple(matrix.shape),
    )

# sample points (element centers, interior) for averaging the axial stress
grid_1d = (np.arange(NELEMENTS) + 0.5) / NELEMENTS * LENGTH
sample_points = [[x, y, z] for x in grid_1d for y in grid_1d for z in grid_1d]

# ----------------------------------- load stepping -----------------------------------
dofs0 = mlhp.DoubleVector([0.0] * ndof)
strain_history, stress_history = [0.0], [0.0]

for istep in range(1, NSTEPS + 1):
    applied = STRAIN_MAX * LENGTH * istep / NSTEPS
    dofs1 = mlhp.DoubleVector(list(dofs0))

    for inewton in range(20):
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
        for face, ifield, target in boundary_conditions(applied):
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

    data = np.array(mlhphelpers.evaluateJ2History(history, sample_points)).reshape(-1, 13)
    axial_stress = data[:, 2].mean()  # S_zz
    strain_history.append(applied / LENGTH)
    stress_history.append(axial_stress)
    print(f"step {istep:2d}: strain {applied / LENGTH:.4f}, axial stress {axial_stress:8.2f}, newton {inewton}")

# ----------------------------------- postprocessing ----------------------------------
strain = np.array(strain_history)
stress = np.array(stress_history)

# analytical uniaxial response: elastic to yield, then tangent E*H/(E+H)
yield_strain = YIELD / E
tangent = E * HARDENING / (E + HARDENING)
analytical = np.where(
    strain <= yield_strain, E * strain, YIELD + tangent * (strain - yield_strain)
)

fig, ax = plt.subplots()
ax.plot(strain, analytical, color="0.6", linestyle="--")
ax.plot(strain, stress, color="#1f77b4", marker="o", markersize=4)
ax.set_xlabel("axial strain")
ax.set_ylabel("axial stress")
fig.tight_layout()

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "j2_plasticity_mlhp.pdf")
else:
    plt.show()
