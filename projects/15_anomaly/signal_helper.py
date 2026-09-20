import math
from typing import Callable

import numpy as np


# --------------------------------------- helper --------------------------------------
def bump(t: float, t0: float, width: float, height: float) -> float:
    """smooth bump of the given height at t0, equal to 1 beyond `width` from t0."""
    if abs(t - t0) < width:
        return (
            math.exp(1.0 / (((t - t0) / width) ** 2 - 1)) * math.exp(1) * (height - 1)
            + 1
        )
    else:
        return 1


def get_degrading_k(
    bump_center: float,
    width: float,
    height: float,
    k_base: np.ndarray,
    idx: int,
) -> Callable:
    """build a stiffness law whose idx-th entry decays to height and stays there."""

    def degrade(t):
        if t < bump_center:
            return bump(t, bump_center, width, height)
        else:
            return height

    def k(t):
        k_mod = k_base.copy()  # assumes k is a fixed vector
        k_mod[idx] *= degrade(t)
        return k_mod

    return k


def get_spiking_f(
    bump_center: float,
    width: float,
    height: float,
    f_base: Callable,
    idx: int,
) -> Callable:
    """build a forcing law whose idx-th component is scaled by a single bump."""
    amp = lambda t: bump(t, bump_center, width, height)
    return lambda t: [
        fi * amp(t) if i == idx else fi for i, fi in enumerate(f_base(t))
    ]
