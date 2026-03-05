import numpy as np


class BouncingBalls:
    def __init__(self, p0, v0, R, bounds, e=0.9, g=None):
        self.p0 = np.array(p0)
        self.v0 = np.array(v0)
        self.R = np.array(R)
        self.m = self.R**2
        self.bounds = bounds
        self.e = e
        self.g = np.array(g) if g is not None else np.array([0, -9.81])

    def solve(self, dt, N):
        balls = len(self.R)
        R, m, e, g = self.R, self.m, self.e, self.g
        bounds = self.bounds

        p = np.zeros((N + 1, balls, 2))
        v = self.v0.copy()
        p[0] = self.p0

        for n in range(N):
            # symplectic Euler
            v = v + g * dt
            p[n + 1] = p[n] + v * dt

            # boundary contact
            left  = p[n + 1, :, 0] < bounds[0][0] + R
            right = p[n + 1, :, 0] > bounds[0][1] - R
            p[n + 1, left,  0] = bounds[0][0] + R[left]
            p[n + 1, right, 0] = bounds[0][1] - R[right]
            v[left,  0] *= -e
            v[right, 0] *= -e

            bot = p[n + 1, :, 1] < bounds[1][0] + R
            top = p[n + 1, :, 1] > bounds[1][1] - R
            p[n + 1, bot, 1] = bounds[1][0] + R[bot]
            p[n + 1, top, 1] = bounds[1][1] - R[top]
            v[bot, 1] *= -e
            v[top, 1] *= -e

            # ball-ball contact
            for i in range(balls):
                for j in range(i + 1, balls):
                    dp = p[n + 1, i] - p[n + 1, j]
                    dist = np.linalg.norm(dp)
                    min_dist = R[i] + R[j]

                    if dist < min_dist:
                        n_ij = dp / (dist + 1e-12)
                        dv = v[i] - v[j]
                        vn = np.dot(dv, n_ij)

                        if vn < 0:
                            mi, mj = m[i], m[j]
                            J = -(1 + e) * vn / (1 / mi + 1 / mj)
                            v[i] += (J / mi) * n_ij
                            v[j] -= (J / mj) * n_ij

                            # positional correction (mass-weighted)
                            overlap = min_dist - dist
                            p[n + 1, i] += overlap * (mj / (mi + mj)) * n_ij
                            p[n + 1, j] -= overlap * (mi / (mi + mj)) * n_ij

        return p
