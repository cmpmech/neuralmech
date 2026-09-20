import random

import numpy as np

# -------------------------------------- settings -------------------------------------
T_BASE = 6.4
DT = 0.05

M0 = np.array([1, 1, 1])
K0 = np.array([2, 2, 2, 2])
D0 = np.array([0.1, 0.1, 0.1, 0.1])
CONNECTIONS = [[None, 0], [0, 1], [1, 2], [2, None]]


# --------------------------------------- helper --------------------------------------
def sample_problem() -> tuple:
    """draw one healthy 3-dof oscillator around the base parameters.

    Returns masses, stiffnesses, dampings, the forcing function, and the initial state.
    """
    m = np.random.uniform(0.9 * M0, 1.1 * M0)
    k = np.random.uniform(0.9 * K0, 1.1 * K0)
    d = np.random.uniform(0.9 * D0, 1.1 * D0)
    u0 = np.random.uniform(-0.1, 0.1, 3)
    du0 = np.random.uniform(-0.1, 0.1, 3)
    freq = random.uniform(0.1, 1)
    amp = random.uniform(0.1, 1)
    shift = random.uniform(0, 1)
    f = lambda t: [0 * t, 0 * t, amp * np.sin(freq * 2 * np.pi * (t + shift))]
    return m, k, d, f, u0, du0
