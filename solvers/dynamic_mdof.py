import numpy as np
import math
from scipy.integrate import odeint

def system(Y, t, m, k, d, f, connections):
    dofs = len(m)
    dudts = Y[dofs:] # transform to first order ODE
    dvdts = f(t) # initialize rhs

    # for time-dependent properties
    if callable(k):
        k = k(t)
    if callable(d):
        d = d(t)
    if callable(m):
        m = d(t)

    for i, (s, r) in enumerate(connections):
        if s is None:
            u_r, v_r = Y[r], Y[dofs + r]
            dvdts[r] -= k[i] * u_r + d[i] * v_r
        elif r is None:
            u_s, v_s = Y[s], Y[dofs + s]
            dvdts[s] -= k[i] * u_s + d[i] * v_s
        else:
            u_s, v_s = Y[s], Y[dofs + s]
            u_r, v_r = Y[r], Y[dofs + r]
            force = k[i] * (u_r - u_s) + d[i] * (v_r - v_s)
            dvdts[s] += force
            dvdts[r] -= force

    dvdts /= m
    return [*dudts, *dvdts]

class MDOF:
    def __init__(self, m, k, d, f, connections):
        self.m, self.f = m, f
        self.k, self.d = k, d
        self.connections = connections

        self.dofs = len(m)

    def solve(self, u0, du0, T, dt=None):
        Y0 = [*u0, *du0]
        if dt is None:
            dt = np.sqrt(np.min(self.m) / np.max(self.k)) * 0.1
        N = int(math.ceil(T / dt))
        t = np.linspace(0, dt * (N - 1), N)

        U = odeint(system, Y0, t, args=(self.m, self.k, self.d, self.f,
                                        self.connections))
        return t, U[:,:self.dofs]