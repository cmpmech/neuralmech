"""Method of Moving Asymptotes (MMA) for topology optimization.

Faithful reimplementation of Svanberg's MMA (Svanberg 1987; the primal-dual
subsolver follows Svanberg's 2007 MATLAB code and Arjen Deetman's Python
port). The math is identical to the ``mmapy`` package, so it is a drop-in
replacement that produces the same iterates, but two things make it fast:

1. mmapy's ``diags(v).dot(M.T).T`` (an n x n sparse matrix built only to scale
   columns) becomes an elementwise product — for a single constraint the
   subproblem is a 2x2 dual solve, so a step is O(n) vector work.
2. The primal-dual interior-point inner loop — which dominates the cost at
   n ~ 1e4 because it touches ~30 length-n temporaries per Newton step — is
   compiled with ``numba``. This removes the Python/dispatch/allocation
   overhead that capped the pure-numpy version at ~1.2x over mmapy.

``mmasub`` keeps mmapy's signature and returns; ``MMA`` wraps the
asymptote/iterate bookkeeping for drivers.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def _accumulate(arr, ssq, rmax):
    for k in range(arr.shape[0]):
        v = arr[k]
        ssq += v * v
        av = v if v >= 0.0 else -v
        if av > rmax:
            rmax = av
    return ssq, rmax


@numba.njit(cache=True)
def _subsolv_kernel(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1.0
    lam = np.ones(m)
    xsi = np.maximum(1.0 / (x - alfa), np.ones(n))
    eta = np.maximum(1.0 / (beta - x), np.ones(n))
    mu = np.maximum(0.5 * c, np.ones(m))
    zet = 1.0
    s = np.ones(m)

    while epsi > epsimin:
        ux1 = upp - x
        xl1 = x - low
        uxinv1 = 1.0 / ux1
        xlinv1 = 1.0 / xl1
        plam = p0.copy()
        qlam = q0.copy()
        for i in range(m):
            plam += P[i] * lam[i]
            qlam += Q[i] * lam[i]
        gvec = np.zeros(m)
        for i in range(m):
            gvec[i] = np.sum(P[i] * uxinv1) + np.sum(Q[i] * xlinv1)
        dpsidx = plam / (ux1 * ux1) - qlam / (xl1 * xl1)
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.sum(a * lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsi
        reeta = eta * (beta - x) - epsi
        remu = mu * y - epsi
        rezet = zet * z - epsi
        res = lam * s - epsi
        ssq = rez * rez + rezet * rezet
        rmax = max(abs(rez), abs(rezet))
        ssq, rmax = _accumulate(rex, ssq, rmax)
        ssq, rmax = _accumulate(rey, ssq, rmax)
        ssq, rmax = _accumulate(relam, ssq, rmax)
        ssq, rmax = _accumulate(rexsi, ssq, rmax)
        ssq, rmax = _accumulate(reeta, ssq, rmax)
        ssq, rmax = _accumulate(remu, ssq, rmax)
        ssq, rmax = _accumulate(res, ssq, rmax)
        residunorm = np.sqrt(ssq)
        residumax = rmax
        ittt = 0

        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            uxinv1 = 1.0 / ux1
            xlinv1 = 1.0 / xl1
            plam = p0.copy()
            qlam = q0.copy()
            for i in range(m):
                plam += P[i] * lam[i]
                qlam += Q[i] * lam[i]
            gvec = np.zeros(m)
            GG = np.empty((m, n))
            for i in range(m):
                gvec[i] = np.sum(P[i] * uxinv1) + np.sum(Q[i] * xlinv1)
                GG[i] = P[i] / ux2 - Q[i] / xl2
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - np.sum(a * lam) - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = 2.0 * (plam / (ux1 * ux2) + qlam / (xl1 * xl2)) + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = 1.0 / diagx
            diagyinv = 1.0 / (d + mu / y)
            diaglamyi = s / lam + diagyinv

            ddx = delx / diagx
            blam = dellam + dely * diagyinv
            for i in range(m):
                blam[i] -= np.sum(GG[i] * ddx)
            Alam = np.zeros((m, m))
            for i in range(m):
                gd = GG[i] * diagxinv
                for k in range(m):
                    Alam[i, k] = np.sum(gd * GG[k])
                Alam[i, i] += diaglamyi[i]
            AA = np.zeros((m + 1, m + 1))
            bb = np.empty(m + 1)
            for i in range(m):
                for k in range(m):
                    AA[i, k] = Alam[i, k]
                AA[i, m] = a[i]
                AA[m, i] = a[i]
                bb[i] = blam[i]
            AA[m, m] = -zet / z
            bb[m] = delz
            solut = np.linalg.solve(AA, bb)
            dlam = solut[0:m]
            dz = solut[m]
            gtdlam = np.zeros(n)
            for i in range(m):
                gtdlam += GG[i] * dlam[i]
            dx = -ddx - gtdlam * diagxinv
            dy = -dely * diagyinv + dlam * diagyinv
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam

            stmxx = -1.01 * dzet / zet
            for i in range(m):
                stmxx = max(stmxx, -1.01 * dy[i] / y[i], -1.01 * dlam[i] / lam[i],
                            -1.01 * dmu[i] / mu[i], -1.01 * ds[i] / s[i])
            stmxx = max(stmxx, -1.01 * dz / z)
            stmalfa = 0.0
            stmbeta = 0.0
            for j in range(n):
                stmxx = max(stmxx, -1.01 * dxsi[j] / xsi[j], -1.01 * deta[j] / eta[j])
                stmalfa = max(stmalfa, -1.01 * dx[j] / (x[j] - alfa[j]))
                stmbeta = max(stmbeta, 1.01 * dx[j] / (beta[j] - x[j]))
            steg = 1.0 / max(stmxx, stmalfa, stmbeta, 1.0)

            xold = x.copy()
            yold = y.copy()
            zold = z
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet
            sold = s.copy()

            itto = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                uxinv1 = 1.0 / ux1
                xlinv1 = 1.0 / xl1
                plam = p0.copy()
                qlam = q0.copy()
                for i in range(m):
                    plam += P[i] * lam[i]
                    qlam += Q[i] * lam[i]
                gvec = np.zeros(m)
                for i in range(m):
                    gvec[i] = np.sum(P[i] * uxinv1) + np.sum(Q[i] * xlinv1)
                dpsidx = plam / (ux1 * ux1) - qlam / (xl1 * xl1)
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - np.sum(a * lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsi
                reeta = eta * (beta - x) - epsi
                remu = mu * y - epsi
                rezet = zet * z - epsi
                res = lam * s - epsi
                ssq = rez * rez + rezet * rezet
                rmax = max(abs(rez), abs(rezet))
                ssq, rmax = _accumulate(rex, ssq, rmax)
                ssq, rmax = _accumulate(rey, ssq, rmax)
                ssq, rmax = _accumulate(relam, ssq, rmax)
                ssq, rmax = _accumulate(rexsi, ssq, rmax)
                ssq, rmax = _accumulate(reeta, ssq, rmax)
                ssq, rmax = _accumulate(remu, ssq, rmax)
                ssq, rmax = _accumulate(res, ssq, rmax)
                resinew = np.sqrt(ssq)
                steg = steg / 2.0
            residunorm = resinew
            residumax = rmax

        epsi = 0.1 * epsi

    return x, y, z, lam, xsi, eta, mu, zet, s


def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    flat = lambda v: np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1))
    x, y, z, lam, xsi, eta, mu, zet, s = _subsolv_kernel(
        int(m), int(n), float(epsimin),
        flat(low), flat(upp), flat(alfa), flat(beta), flat(p0), flat(q0),
        np.ascontiguousarray(np.asarray(P, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(Q, dtype=np.float64)),
        float(a0), flat(a), flat(b), flat(c), flat(d))
    col = lambda v: v.reshape(-1, 1)
    return (col(x), col(y), np.array([[z]]), col(lam), col(xsi), col(eta),
            col(mu), np.array([[zet]]), col(s))


def _subproblem(m, n, iter, xval, xmin, xmax, xold1, xold2, df0dx, dfdx, low, upp,
                move, asyinit, asydecr, asyincr, asymin, asymax, raa0, albefa, fval):
    # update moving asymptotes (low/upp), trust bounds (alfa/beta) and build the
    # convex separable MMA approximation coefficients p0/q0/P/Q and the RHS b
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    if iter <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        low = np.minimum(np.maximum(low, xval - asymax * (xmax - xmin)), xval - asymin * (xmax - xmin))
        upp = np.maximum(np.minimum(upp, xval + asymax * (xmax - xmin)), xval + asymin * (xmax - xmin))

    alfa = np.maximum(np.maximum(low + albefa * (xval - low), xval - move * (xmax - xmin)), xmin)
    beta = np.minimum(np.minimum(upp - albefa * (upp - xval), xval + move * (xmax - xmin)), xmax)

    xmami_inv = eeen / np.maximum(xmax - xmin, 1e-5 * eeen)
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1

    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    P = np.maximum(dfdx, 0)
    Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.dot(eeem, xmami_inv.T)
    P = (P + PQ) * ux2.flatten()   # was diags(ux2).dot(P.T).T
    Q = (Q + PQ) * xl2.flatten()
    b = np.dot(P, eeen / ux1) + np.dot(Q, eeen / xl1) - fval
    return low, upp, alfa, beta, p0, q0, P, Q, b


def _dual_solve(low, upp, alfa, beta, p0, q0, P, Q, b):
    # single-constraint MMA: minimize the separable approximation over x in [alfa, beta]
    # for a trial multiplier lam, x(lam) is closed-form; bisect lam >= 0 to satisfy the
    # one constraint g(lam) = sum_j[P_j/(upp-x) + Q_j/(x-low)] - b <= 0 (= 0 if active).
    # This is Svanberg's dual, exact whenever the constraint is feasible (artificial y,z
    # stay zero) -- the regime of a volume constraint. Falls back to interior for m > 1.
    p0 = p0.ravel(); q0 = q0.ravel(); P = P.ravel(); Q = Q.ravel()
    low = low.ravel(); upp = upp.ravel(); alfa = alfa.ravel(); beta = beta.ravel()
    b = float(np.ravel(b)[0])

    def xlam(lam):
        sp = np.sqrt(p0 + lam * P)
        sq = np.sqrt(q0 + lam * Q)
        return np.clip((sp * low + sq * upp) / (sp + sq), alfa, beta)

    def g(lam):
        x = xlam(lam)
        return np.sum(P / (upp - x) + Q / (x - low)) - b

    if g(0.0) <= 0.0:
        return xlam(0.0).reshape(-1, 1)
    hi = 1.0
    while g(hi) > 0.0 and hi < 1e16:
        hi *= 2.0
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return xlam(0.5 * (lo + hi)).reshape(-1, 1)


def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx,
           low, upp, a0, a, c, d, move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2,
           asymin=0.01, asymax=10.0, raa0=1e-5, albefa=0.1):
    low, upp, alfa, beta, p0, q0, P, Q, b = _subproblem(
        m, n, iter, xval, xmin, xmax, xold1, xold2, df0dx, dfdx, low, upp,
        move, asyinit, asydecr, asyincr, asymin, asymax, raa0, albefa, fval)
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, 1e-7, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def mma_dual(n, iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp,
             move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2,
             asymin=0.01, asymax=10.0, raa0=1e-5, albefa=0.1):
    low, upp, alfa, beta, p0, q0, P, Q, b = _subproblem(
        1, n, iter, xval, xmin, xmax, xold1, xold2, df0dx, dfdx, low, upp,
        move, asyinit, asydecr, asyincr, asymin, asymax, raa0, albefa, fval)
    return _dual_solve(low, upp, alfa, beta, p0, q0, P, Q, b), low, upp


class MMA:
    """Stateful wrapper around the MMA subproblem solve for SIMP drivers.

    Holds the asymptote history (``low``/``upp``/``xold1``/``xold2``) across
    the whole run so a beta-continuation loop keeps MMA's memory, and exposes
    a single ``step`` taking the current objective/constraint values and
    gradients. ``m`` constraints, ``n`` design variables; ``move`` is the
    per-step bound (default 0.2, the structural-optimization value).

    ``solver="interior"`` (default) uses the general-``m`` primal-dual kernel
    that reproduces mmapy's iterates -- and hence its optimum. ``solver="dual"``
    uses the fast single-constraint dual, which solves each subproblem but takes
    a different trajectory and tends to a slightly worse (non-convex) optimum, so
    it is opt-in. ``m > 1`` always uses the interior solver.
    """

    def __init__(self, n, m=1, xmin=0.0, xmax=1.0, move=0.2, a0=1.0, a=None, c=1e3,
                 d=0.0, solver="interior"):
        self.n = int(n)
        self.m = int(m)
        self.move = move
        self.solver = solver if self.m == 1 else "interior"
        self.xmin = np.full((self.n, 1), xmin) if np.isscalar(xmin) else xmin.reshape(self.n, 1)
        self.xmax = np.full((self.n, 1), xmax) if np.isscalar(xmax) else xmax.reshape(self.n, 1)
        self.a0 = a0
        self.a = np.zeros((self.m, 1)) if a is None else np.full((self.m, 1), a)
        self.c = np.full((self.m, 1), c)
        self.d = np.full((self.m, 1), d)
        self.low = self.xmin.copy()
        self.upp = self.xmax.copy()
        self.xold1 = None
        self.xold2 = None
        self.iter = 0

    def step(self, xval, f0val, df0dx, fval, dfdx):
        xval = xval.reshape(self.n, 1)
        if self.xold1 is None:
            self.xold1 = xval.copy()
            self.xold2 = xval.copy()
        self.iter += 1
        df0dx = df0dx.reshape(self.n, 1)
        fval = np.atleast_2d(np.asarray(fval, dtype=float)).reshape(self.m, 1)
        dfdx = dfdx.reshape(self.m, self.n)
        if self.solver == "dual":
            xmma, self.low, self.upp = mma_dual(
                self.n, self.iter, xval, self.xmin, self.xmax, self.xold1, self.xold2,
                df0dx, fval, dfdx, self.low, self.upp, move=self.move)
        else:
            xmma, _, _, _, _, _, _, _, _, self.low, self.upp = mmasub(
                self.m, self.n, self.iter, xval, self.xmin, self.xmax, self.xold1, self.xold2,
                f0val, df0dx, fval, dfdx, self.low, self.upp, self.a0, self.a, self.c, self.d,
                move=self.move)
        self.xold2 = self.xold1
        self.xold1 = xval
        return xmma
