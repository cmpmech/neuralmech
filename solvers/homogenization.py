"""linear-elastic RVE homogenization by the finite cell method in mlhp."""

import mlhp
import numpy as np

# Voigt component -> (i, j) of the strain tensor, engineering shear on off-diagonals.
# Ordering matches mlhp's VoigtIndices (2D [11, 22, 12], 3D [11, 22, 33, 12, 23, 13]).
_VOIGT = {
    2: [(0, 0), (1, 1), (0, 1)],
    3: [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)],
}


def effective_stiffness(basis, quadrature, kinematics, material, lengths) -> np.ndarray:
    """apparent KUBC stiffness of the cell discretized by `basis`, in Voigt notation.

    Each entry is read off the internal energy of a unit strain state imposed as a
    linear Dirichlet displacement on all faces.

    Args:
        basis: mlhp basis on the background grid covering the cell.
        quadrature: cut-cell quadrature resolving the material interface.
        kinematics: `mlhp.smallStrainKinematics` of the cell dimension.
        material: mlhp material with the cell's spatially varying coefficients.
        lengths: side lengths of the cell; set the dimension and the volume.

    Returns:
        symmetric (3, 3) in 2D or (6, 6) in 3D matrix with engineering shear.
    """
    dimensions = len(lengths)
    volume = float(np.prod(lengths))
    voigt = _VOIGT[dimensions]
    ncomp = len(voigt)
    coords = ["x", "y", "z"][:dimensions]
    faces = list(range(2 * dimensions))

    integrand = mlhp.staticDomainIntegrand(kinematics, material)

    def internal_energy(strain) -> float:
        # symmetric strain tensor from the Voigt vector (halve the shear entries),
        # imposed as the linear KUBC displacement u_i = sum_j eps_ij x_j
        eps = np.zeros((dimensions, dimensions))
        for c, (i, j) in enumerate(voigt):
            eps[i, j] = eps[j, i] = strain[c] / (1 if i == j else 2)
        u = [
            "+".join(f"{eps[i, j]}*{coords[j]}" for j in range(dimensions))
            for i in range(dimensions)
        ]
        dofs = mlhp.integrateDirichletDofs(
            mlhp.vectorField(dimensions, "[" + ",".join(u) + "]"), basis, faces
        )

        matrix = mlhp.allocateSparseMatrix(basis, dofs[0])
        vector = mlhp.allocateRhsVector(matrix)
        mlhp.integrateOnDomain(
            basis,
            integrand,
            [matrix, vector],
            quadrature=quadrature,
            dirichletDofs=dofs,
        )
        interior = mlhp.makeCGSolver(rtol=1e-12, maxiter=10000)(matrix, vector)
        allDofs = mlhp.inflateDofs(interior, dofs)

        energy = mlhp.ScalarDouble(0.0)
        energyIntegrand = mlhp.internalEnergyIntegrand(allDofs, kinematics, material)
        mlhp.integrateOnDomain(basis, energyIntegrand, [energy], quadrature=quadrature)
        return 2.0 * energy.get() / volume

    diagonal = [internal_energy(row) for row in np.eye(ncomp)]

    C = np.diag(diagonal)
    for i in range(ncomp):
        for j in range(i + 1, ncomp):
            C[i, j] = C[j, i] = 0.5 * (
                internal_energy(np.eye(ncomp)[i] + np.eye(ncomp)[j])
                - diagonal[i]
                - diagonal[j]
            )
    return C
