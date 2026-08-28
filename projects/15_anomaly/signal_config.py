import random

import numpy as np

# Base problem parameters
T_base, dt = 6.4, 0.05

m0 = np.array([1, 1, 1])
k0 = np.array([2, 2, 2, 2])
d0 = np.array([0.1, 0.1, 0.1, 0.1])
connections = [[None, 0], [0, 1], [1, 2], [2, None]]


def sample_problem():
    m = np.random.uniform(0.9 * m0, 1.1 * m0)
    k = np.random.uniform(0.9 * k0, 1.1 * k0)
    d = np.random.uniform(0.9 * d0, 1.1 * d0)
    u0 = np.random.uniform(-0.1, 0.1, 3)
    du0 = np.random.uniform(-0.1, 0.1, 3)
    freq = random.uniform(0.1, 1)  # random.uniform(0.5, 2)
    amp = random.uniform(0.1, 1)
    shift = random.uniform(0, 1)
    f = lambda t: [0 * t, 0 * t, amp * np.sin(freq * 2 * np.pi * (t + shift))]
    return m, k, d, f, u0, du0
