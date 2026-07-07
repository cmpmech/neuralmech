from pathlib import Path

import mlhp
import numpy as np
from tqdm import tqdm

from solvers.homogenization import effective_stiffness
from solvers.material_subroutines.nonlinear_elastic import ABI, build

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 300

# discretization
DIM = 2
DEGREE = 2
NELEMENTS = [12] * DIM

# microstructure: centered circular phase-2 inclusion in a phase-1 matrix
LENGTH = 1.0
RADIUS = 0.25

# phase sampling: log-uniform moduli with a soft-but-not-void inclusion (moderate
# contrast keeps the homogenization well conditioned), moderate Poisson ratios
E1_RANGE = (0.8, 1.25)
E2_RANGE = (0.1, 1.0)
NU_RANGE = (0.15, 0.35)

# nonlinear-elastic reference: sigma = C0 eps + c (eps.eps) eps per phase (E, nu, c), driven
# along a uniaxial macro-strain path; the same law is applied online in dmn_exampleV2.py
MATRIX = (1.0, 0.3, 30.0)
INCLUSION = (0.2, 0.3, 10.0)
EPS_MAX = 0.2
NSTEPS = 50
NEWTON_ITER = 25
NEWTON_TOL = 1e-8


# ------------------------------------ helper -----------------------------------------
def isotropic_stiffness(E, nu):
    return E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


# ----------------------------------- create data -------------------------------------
# fixed RVE (geometry, mesh, quadrature are material independent, so build them once)
lengths = [LENGTH] * DIM
center = [0.5 * LENGTH] * DIM
inside = f"((x - {center[0]})**2 + (y - {center[1]})**2) < {RADIUS**2}"

inclusion = mlhp.implicitSphere(center, RADIUS)
grid = mlhp.makeRefinedGrid(NELEMENTS, lengths)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=DIM)
kinematics = mlhp.smallStrainKinematics(DIM)
quadrature = mlhp.spaceTreeQuadrature(inclusion, depth=DEGREE + 1, epsilon=1.0)

E1 = 10 ** rng.uniform(np.log10(E1_RANGE[0]), np.log10(E1_RANGE[1]), SAMPLES)
E2 = 10 ** rng.uniform(np.log10(E2_RANGE[0]), np.log10(E2_RANGE[1]), SAMPLES)
nu1 = rng.uniform(*NU_RANGE, SAMPLES)
nu2 = rng.uniform(*NU_RANGE, SAMPLES)

C1 = np.zeros((SAMPLES, 3, 3))
C2 = np.zeros((SAMPLES, 3, 3))
C_eff = np.zeros((SAMPLES, 3, 3))

for i in tqdm(range(SAMPLES)):
    C1[i] = isotropic_stiffness(E1[i], nu1[i])
    C2[i] = isotropic_stiffness(E2[i], nu2[i])
    E = mlhp.scalarField(DIM, f"{E2[i]} if {inside} else {E1[i]}")
    nu = mlhp.scalarField(DIM, f"{nu2[i]} if {inside} else {nu1[i]}")
    material = mlhp.planeStressMaterial(E, nu)
    C_eff[i] = effective_stiffness(basis, quadrature, kinematics, material, lengths)

# --------------------------------- reference solution --------------------------------
# genuine nonlinear FE homogenization of the same RVE: both phases follow the nonlinear
# law (cffi subroutine), KUBC uniaxial macro strain [e11, 0, 0] on all faces, Newton per
# step; macro stress s11 = -(reactions . unit-e11 displacement pattern) / volume
lib = build()
faces = list(range(2 * DIM))
volume = float(np.prod(lengths))

matrix_law = mlhp.constitutiveEquation(
    DIM,
    lib.material_address,
    symmetric=True,
    incremental=False,
    data=[list(MATRIX)],
    abi=ABI,
)
inclusion_law = mlhp.constitutiveEquation(
    DIM,
    lib.material_address,
    symmetric=True,
    incremental=False,
    data=[list(INCLUSION)],
    abi=ABI,
)


def internal_force(u, dirichlet):
    integrand = mlhp.selectIntegrand(
        inclusion,
        mlhp.staticDomainIntegrand(kinematics, inclusion_law, dofs=u),
        default=mlhp.staticDomainIntegrand(kinematics, matrix_law, dofs=u),
    )
    K = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(K)
    mlhp.integrateOnDomain(
        basis, integrand, [K, vector], quadrature=quadrature, dirichletDofs=dirichlet
    )
    return K, vector


# work-conjugate pattern for the macro stress: the unit-e11 KUBC displacement u = [x, 0]
unconstrained = mlhp.combineDirichletDofs([])
mode = mlhp.integrateDirichletDofs(mlhp.vectorField(DIM, "[x, 0]"), basis, faces)
mode_idx, mode_val = np.array(mode[0]), np.array(mode[1])

ref_eps, ref_sig = [0.0], [0.0]
for step in tqdm(range(1, NSTEPS + 1)):
    e11 = EPS_MAX * step / NSTEPS
    dirichlet = mlhp.combineDirichletDofs(
        [
            mlhp.integrateDirichletDofs(
                mlhp.vectorField(DIM, f"[{e11} * x, 0]"), basis, faces
            )
        ]
    )
    zero = [dirichlet[0], mlhp.DoubleVector(len(dirichlet[0]), 0.0)]
    u = mlhp.inflateDofs(
        mlhp.DoubleVector(basis.ndof() - len(dirichlet[0]), 0.0), dirichlet
    )

    for it in range(NEWTON_ITER):
        K, residual = internal_force(u, zero)
        norm = mlhp.norm(residual)
        reference = norm if it == 0 else reference
        if norm <= max(NEWTON_TOL * reference, 1e-14):
            break
        du = mlhp.cg(
            K, residual, M=mlhp.diagonalPreconditioner(K), rtol=1e-12, maxiter=10000
        )
        u = mlhp.add(u, mlhp.inflateDofs(du, zero))

    _, fint = internal_force(u, unconstrained)
    ref_eps.append(e11)
    ref_sig.append(-float(np.sum(np.array(fint)[mode_idx] * mode_val)) / volume)

# ------------------------------------- export ----------------------------------------
np.savez(
    DATA_DIR / "hom_dmn.npz",
    C1=C1,
    C2=C2,
    C_eff=C_eff,
    ref_eps=np.array(ref_eps),
    ref_sig=np.array(ref_sig),
    matrix=np.array(MATRIX),
    inclusion=np.array(INCLUSION),
    eps_max=EPS_MAX,
    nsteps=NSTEPS,
)
print(f"saved {SAMPLES} samples + nonlinear reference to {DATA_DIR / 'hom_dmn.npz'}")
