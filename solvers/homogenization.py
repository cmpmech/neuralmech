"""RVE homogenization of a unit square with a centered circular inclusion of
phase 2 in a phase-1 matrix, computed with the finite cell method in mlhp.

``effective_stiffness`` returns the linear apparent stiffness under kinematic
uniform boundary conditions (KUBC): for a macro strain ``eps`` (Voigt
``[11, 22, 12]`` with engineering shear), imposing ``u = eps . x`` on the whole
boundary gives ``U = 0.5 * V * eps^T C_eff eps``, so the three unit strains and
their three pairwise sums recover the full symmetric 3x3 ``C_eff`` (here the cell
volume ``V`` and the strain amplitude are both 1, so ``eps^T C_eff eps = 2 U``).

``nonlinear_uniaxial_response`` drives the same cell with a user nonlinear-elastic
mlhp material subroutine along a uniaxial KUBC macro-strain path and returns the
volume-averaged macro stress-strain curve, as a reference for a deep material
network trained on the linear data.
"""

import mlhp
import numpy as np

# KUBC boundary displacement u(x) = eps . x for each probed macro strain, as
# [ux, uy]; strains are Voigt [e11, e22, g12] with engineering shear g12.
_FIELDS = ["[x, 0]", "[0, y]", "[y, 0]", "[x, y]", "[x + y, 0]", "[y, y]"]


def effective_stiffness(
    E1: float,
    nu1: float,
    E2: float,
    nu2: float,
    radius: float = 0.25,
    nelements: int = 16,
    degree: int = 2,
) -> np.ndarray:
    """Return the 3x3 plane-stress stiffness (Voigt [11, 22, 12]) of a unit cell
    with a phase-2 circular inclusion of the given radius in a phase-1 matrix."""
    inclusion = mlhp.implicitSphere([0.5, 0.5], radius)
    inside = f"((x - 0.5)**2 + (y - 0.5)**2) < {radius * radius}"
    E = mlhp.scalarField(2, f"{E2} if {inside} else {E1}")
    nu = mlhp.scalarField(2, f"{nu2} if {inside} else {nu1}")

    grid = mlhp.makeRefinedGrid([nelements, nelements], [1.0, 1.0])
    basis = mlhp.makeHpTrunkSpace(grid, degree=degree, nfields=2)

    kinematics = mlhp.smallStrainKinematics(2)
    material = mlhp.planeStressMaterial(E, nu)
    integrand = mlhp.staticDomainIntegrand(kinematics, material)
    # space tree (not moment fitting) resolves the material jump at the interface
    quadrature = mlhp.spaceTreeQuadrature(inclusion, depth=degree + 1, epsilon=1.0)

    energies = []
    for field in _FIELDS:
        dofs = mlhp.integrateDirichletDofs(mlhp.vectorField(2, field), basis, [0, 1, 2, 3])
        matrix = mlhp.allocateSparseMatrix(basis, dofs[0])
        vector = mlhp.allocateRhsVector(matrix)
        mlhp.integrateOnDomain(
            basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dofs
        )
        interior = mlhp.makeCGSolver(rtol=1e-12, maxiter=10000)(matrix, vector)
        allDofs = mlhp.inflateDofs(interior, dofs)

        energy = mlhp.ScalarDouble(0.0)
        energyIntegrand = mlhp.internalEnergyIntegrand(allDofs, kinematics, material)
        mlhp.integrateOnDomain(basis, energyIntegrand, [energy], quadrature=quadrature)
        energies.append(2.0 * energy.get())

    C = np.zeros((3, 3))
    C[0, 0], C[1, 1], C[2, 2] = energies[0], energies[1], energies[2]
    C[0, 1] = C[1, 0] = 0.5 * (energies[3] - C[0, 0] - C[1, 1])
    C[0, 2] = C[2, 0] = 0.5 * (energies[4] - C[0, 0] - C[2, 2])
    C[1, 2] = C[2, 1] = 0.5 * (energies[5] - C[1, 1] - C[2, 2])
    return C


def nonlinear_uniaxial_response(
    material_address: int,
    matrix_params: list,
    inclusion_params: list,
    abi: int,
    eps_max: float = 0.08,
    nsteps: int = 50,
    radius: float = 0.25,
    nelements: int = 12,
    degree: int = 2,
    newton_iter: int = 25,
    newton_tol: float = 1e-8,
) -> np.ndarray:
    """Return the uniaxial macro stress-strain curve (shape (nsteps, 2), columns
    e11 and s11) of the two-phase cell under a KUBC path eps = [e11, 0, 0], with a
    user nonlinear-elastic mlhp material subroutine (matrix and inclusion parameter
    vectors, registered incremental=False). The macro stress is the volume average
    <s11>, recovered from the boundary reactions work-conjugate to the unit e11
    displacement pattern (V<s11> = -reactions . mode)."""
    inclusion = mlhp.implicitSphere([0.5, 0.5], radius)
    grid = mlhp.makeRefinedGrid([nelements, nelements], [1.0, 1.0])
    basis = mlhp.makeHpTrunkSpace(grid, degree=degree, nfields=2)
    kinematics = mlhp.smallStrainKinematics(2)
    quadrature = mlhp.spaceTreeQuadrature(inclusion, depth=degree + 1, epsilon=1.0)

    material_m = mlhp.constitutiveEquation(
        2, material_address, symmetric=True, incremental=False, data=[matrix_params], abi=abi
    )
    material_i = mlhp.constitutiveEquation(
        2, material_address, symmetric=True, incremental=False, data=[inclusion_params], abi=abi
    )

    def internal_force(u, dirichlet):
        integrand = mlhp.selectIntegrand(
            inclusion,
            mlhp.staticDomainIntegrand(kinematics, material_i, dofs=u),
            default=mlhp.staticDomainIntegrand(kinematics, material_m, dofs=u),
        )
        matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
        vector = mlhp.allocateRhsVector(matrix)
        mlhp.integrateOnDomain(
            basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
        )
        return matrix, vector

    unconstrained = mlhp.combineDirichletDofs([])
    mode = mlhp.integrateDirichletDofs(mlhp.vectorField(2, "[x, 0]"), basis, [0, 1, 2, 3])
    mode_idx, mode_val = np.array(mode[0]), np.array(mode[1])

    curve = []
    for step in range(1, nsteps + 1):
        eps = eps_max * step / nsteps
        dirichlet = mlhp.combineDirichletDofs(
            [mlhp.integrateDirichletDofs(mlhp.vectorField(2, f"[{eps}*x, 0]"), basis, [0, 1, 2, 3])]
        )
        zero = [dirichlet[0], mlhp.DoubleVector(len(dirichlet[0]), 0.0)]
        u = mlhp.inflateDofs(mlhp.DoubleVector(basis.ndof() - len(dirichlet[0]), 0.0), dirichlet)

        for inewton in range(newton_iter):
            matrix, vector = internal_force(u, zero)
            residual = mlhp.norm(vector)
            reference = residual if inewton == 0 else reference
            if residual <= max(newton_tol * reference, 1e-14):
                break
            preconditioner = mlhp.diagonalPreconditioner(matrix)
            increment = mlhp.cg(matrix, vector, M=preconditioner, rtol=1e-12, maxiter=10000)
            u = mlhp.add(u, mlhp.inflateDofs(increment, zero))

        _, fint = internal_force(u, unconstrained)
        sigma11 = -float(np.sum(np.array(fint)[mode_idx] * mode_val))
        curve.append((eps, sigma11))
    return np.array(curve)
