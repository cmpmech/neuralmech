import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def _rotation(angle):
    """2D rotation matrix for a counter-clockwise angle (sympy expression)."""
    return sp.Matrix([[sp.cos(angle), -sp.sin(angle)], [sp.sin(angle), sp.cos(angle)]])


class PlanarMultibody:
    """planar rigid-body dynamics of an open kinematic tree.

    Each body carries one generalized coordinate, a revolute angle or a prismatic
    slide relative to its parent. The Lagrangian is built symbolically from the forward
    kinematics and reduced to `M(q) qdd = b(q, qd) + Q`, then lambdified to NumPy and
    integrated with scipy. A triple pendulum is three revolute bodies in a chain; a
    cart-pole is a prismatic cart carrying a revolute pole.

    Args:
        bodies: body dicts in topological order (parents first) with keys `parent`
            (index, or None for ground), `joint` ("revolute" or "prismatic"), `anchor`
            (joint location in the parent frame), `axis` (slide direction in the
            parent frame, prismatic only), `com` (offset from the joint in the body
            frame), `mass`, and `inertia` (about the COM, 0 for a point mass).
        g: gravitational acceleration along -y.
    """

    def __init__(self, bodies: list[dict], g: float = 9.81):
        n = len(bodies)
        self.n = n
        self.bodies = bodies

        q = sp.Matrix(sp.symbols(f"q0:{n}", real=True))
        qd = sp.Matrix(sp.symbols(f"qd0:{n}", real=True))
        qdd = sp.Matrix(sp.symbols(f"qdd0:{n}", real=True))

        origins = [None] * n  # inboard-joint location of each body (world frame)
        phis = [None] * n  # absolute orientation of each body
        coms = [None] * n  # center-of-mass location of each body (world frame)

        T = sp.Integer(0)
        V = sp.Integer(0)

        for i, body in enumerate(bodies):
            parent = body["parent"]
            if parent is None:
                origin_parent = sp.Matrix([0, 0])
                phi_parent = sp.Integer(0)
            else:
                origin_parent = origins[parent]
                phi_parent = phis[parent]

            anchor = sp.Matrix(body["anchor"])
            origin = origin_parent + _rotation(phi_parent) * anchor

            if body["joint"] == "revolute":
                phi = phi_parent + q[i]
            elif body["joint"] == "prismatic":
                axis = sp.Matrix(body["axis"])
                origin = origin + q[i] * _rotation(phi_parent) * axis
                phi = phi_parent
            else:
                raise ValueError(f"unknown joint type {body['joint']!r}")

            com = origin + _rotation(phi) * sp.Matrix(body["com"])
            origins[i] = origin
            phis[i] = phi
            coms[i] = com

            # velocities via the kinematic jacobians: v = (dp/dq) qd, w = (dphi/dq) qd
            v = com.jacobian(q) * qd
            w = sp.Matrix([phi]).jacobian(q) * qd

            inertia = body.get("inertia", 0.0)
            T += sp.Rational(1, 2) * body["mass"] * v.dot(v)
            T += sp.Rational(1, 2) * inertia * w[0] ** 2
            V += body["mass"] * g * com[1]

        # Euler-Lagrange: d/dt(dL/dqd) - dL/dq = Q, linear in qdd
        L = T - V
        dL_dqd = sp.Matrix([sp.diff(L, qd[j]) for j in range(n)])
        ddt_dL_dqd = dL_dqd.jacobian(q) * qd + dL_dqd.jacobian(qd) * qdd
        dL_dq = sp.Matrix([sp.diff(L, q[j]) for j in range(n)])
        euler_lagrange = ddt_dL_dqd - dL_dq

        # split into M(q) qdd = b(q, qd): everything moved so the rhs is the bias
        M_sym, b_sym = sp.linear_eq_to_matrix(list(euler_lagrange), list(qdd))

        q_syms, qd_syms = list(q), list(qd)
        self._mass_matrix = sp.lambdify((q_syms, qd_syms), M_sym, "numpy")
        self._bias = sp.lambdify((q_syms, qd_syms), b_sym, "numpy")
        self._origins = sp.lambdify(
            (q_syms,), sp.Matrix.vstack(*[o.T for o in origins]), "numpy"
        )
        self._coms = sp.lambdify(
            (q_syms,), sp.Matrix.vstack(*[c.T for c in coms]), "numpy"
        )

    def solve(self, q0, qd0, T: float, dt: float, Q=None):
        """integrate the equations of motion from (q0, qd0) to time T.

        Returns t of shape (steps,) and q of shape (steps, n).

        Args:
            dt: spacing of the returned samples.
            Q: optional generalized forces `Q(t, q, qd) -> (n,)`, e.g. actuation or
                joint damping; for a prismatic cart this is the force on the cart.
        """
        q0 = np.asarray(q0, dtype=float)
        qd0 = np.asarray(qd0, dtype=float)
        if Q is None:
            Q = lambda t, q, qd: np.zeros(self.n)

        def rhs(t, y):
            q, qd = y[: self.n], y[self.n :]
            M = np.asarray(self._mass_matrix(q, qd), dtype=float).reshape(
                self.n, self.n
            )
            b = np.asarray(self._bias(q, qd), dtype=float).reshape(self.n)
            qdd = np.linalg.solve(M, b + np.asarray(Q(t, q, qd), dtype=float))
            return np.concatenate([qd, qdd])

        steps = int(np.ceil(T / dt))
        t_eval = np.linspace(0, dt * steps, steps + 1)
        sol = solve_ivp(
            rhs,
            (0, t_eval[-1]),
            np.concatenate([q0, qd0]),
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-10,
            atol=1e-10,
        )
        return sol.t, sol.y[: self.n].T

    def forward_kinematics(self, q):
        """world-frame joint origins and centers of mass, each of shape (n, 2)."""
        q = np.asarray(q, dtype=float)
        origins = np.asarray(self._origins(q), dtype=float).reshape(self.n, 2)
        coms = np.asarray(self._coms(q), dtype=float).reshape(self.n, 2)
        return origins, coms
