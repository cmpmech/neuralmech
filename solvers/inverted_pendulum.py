import numpy as np
import math
from scipy.integrate import odeint

def system(Y, t, M, m, l, g, f, d_cart, d_pole):
    x, th, dx, dth = Y
    u = f(t, Y) # state-aware control input (force on cart)

    s, c = np.sin(th), np.cos(th)

    # equations of motion, written as A [ddx, ddth] = b
    # (M + m) ddx + m l cos(th) ddth - m l sin(th) dth^2 = u - d_cart dx
    # m l cos(th) ddx + m l^2 ddth - m g l sin(th)       = -d_pole dth
    A = np.array([[M + m,       m * l * c],
                  [m * l * c,   m * l**2]])
    b = np.array([u - d_cart * dx + m * l * s * dth**2,
                  m * g * l * s - d_pole * dth])

    ddx, ddth = np.linalg.solve(A, b)
    return [dx, dth, ddx, ddth]

class CartPole:
    def __init__(self, M, m, l, g, f, d_cart=0.0, d_pole=0.0):
        self.M, self.m, self.l = M, m, l
        self.g, self.f = g, f
        self.d_cart, self.d_pole = d_cart, d_pole

    def solve(self, Y0, T, dt):
        N = int(math.ceil(T / dt))
        t = np.linspace(0, dt * (N - 1), N)

        Y = odeint(system, Y0, t, args=(self.M, self.m, self.l, self.g,
                                        self.f, self.d_cart, self.d_pole))
        return t, Y
