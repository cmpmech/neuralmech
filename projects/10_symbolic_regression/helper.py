from collections import Counter
from itertools import combinations_with_replacement

import numpy as np


def stlsq(
    theta: np.ndarray, target: np.ndarray, threshold: float, iters: int = 10
) -> np.ndarray:
    """sequentially thresholded least squares.

    Solves target = theta @ xi for a sparse xi: coefficients below `threshold` are set
    to zero and the remaining ones are refitted, `iters` times.

    Reference: https://doi.org/10.1073/pnas.1517384113

    Args:
        theta: (N, terms) library evaluated at the samples.
        target: (N, outputs) quantity each output column is regressed on.
    Returns:
        (terms, outputs) coefficients.
    """
    xi = np.linalg.lstsq(theta, target, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < threshold
        xi[small] = 0
        for j in range(target.shape[1]):
            big = ~small[:, j]
            xi[big, j] = np.linalg.lstsq(theta[:, big], target[:, j], rcond=None)[0]
    return xi


def polynomial_library(
    X: np.ndarray, degree: int, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """monomials of the columns of X up to `degree`; returns (library, term names)."""
    columns = [np.ones(len(X))]
    terms = ["1"]
    for d in range(1, degree + 1):
        for ids in combinations_with_replacement(range(X.shape[1]), d):
            columns.append(np.prod(X[:, ids], axis=1))
            powers = Counter(names[i] for i in ids)
            terms.append(
                " ".join(n if p == 1 else f"{n}^{p}" for n, p in powers.items())
            )
    return np.stack(columns, axis=1), terms
