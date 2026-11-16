import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import factorized


# ------------------------------- beam finite elements --------------------------------
def beam_matrices(EI: np.ndarray) -> tuple:
    """stiffness and consistent mass of a unit cantilever of Hermite beam elements.

    Each node carries deflection and rotation; the clamped node at x = 0 is removed,
    so dof 2 i - 2 is the deflection of node i. Unit length and mass per length.

    Args:
        EI: bending stiffness of each element, its length sets the element count.
    """
    E = len(EI)
    h = 1 / E
    ke = np.array(
        [
            [12, 6 * h, -12, 6 * h],
            [6 * h, 4 * h**2, -6 * h, 2 * h**2],
            [-12, -6 * h, 12, -6 * h],
            [6 * h, 2 * h**2, -6 * h, 4 * h**2],
        ]
    ) / h**3
    me = h / 420 * np.array(
        [
            [156, 22 * h, 54, -13 * h],
            [22 * h, 4 * h**2, 13 * h, -3 * h**2],
            [54, 13 * h, 156, -22 * h],
            [-13 * h, -3 * h**2, -22 * h, 4 * h**2],
        ]
    )
    dofs = 2 * np.arange(E)[:, None] + np.arange(4)
    rows = np.repeat(dofs, 4, axis=1).flatten()
    cols = np.tile(dofs, 4).flatten()
    n = 2 * (E + 1)
    K = coo_matrix((np.outer(EI, ke).flatten(), (rows, cols)), shape=(n, n)).tocsr()
    M = coo_matrix((np.tile(me.flatten(), E), (rows, cols)), shape=(n, n)).tocsr()
    return K[2:, 2:], M[2:, 2:]


# ----------------------------------- time stepping -----------------------------------
def newmark(M, K, shape: np.ndarray, amplitude: np.ndarray, dt: float) -> np.ndarray:
    """average acceleration Newmark integration from rest; returns (steps, dofs).

    The load at step tau is amplitude[tau] * shape. Sparse matrices (full model) are
    factorized once, dense ones (reduced model) are inverted once.
    """
    K_eff = K + 4 / dt**2 * M
    solve = factorized(K_eff.tocsc()) if issparse(K_eff) else np.linalg.inv(K_eff).dot
    u = np.zeros(len(shape))
    v = np.zeros(len(shape))
    a = np.zeros(len(shape))
    U = np.zeros((len(amplitude), len(shape)))
    for tau in range(1, len(amplitude)):
        u_new = solve(amplitude[tau] * shape + M @ (4 / dt**2 * u + 4 / dt * v + a))
        a_new = 4 / dt**2 * (u_new - u) - 4 / dt * v - a
        v = v + dt / 2 * (a + a_new)
        u, a = u_new, a_new
        U[tau] = u
    return U
