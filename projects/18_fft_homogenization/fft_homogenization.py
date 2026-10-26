"""FFT-based homogenization of a periodic 2D cell in plane strain.

Strains and stresses are stored as tensor components (xx, yy, xy) on an N x N pixel grid,
the three unit load cases are batched in front: arrays of shape (3, 3, N, N). The same
code runs on the CPU (numpy, scipy.fft) and on the GPU (cupy, cuFFT): `xp` selects.
"""

import cupy as cp
import cupyx.scipy.fft
import numpy as np
import scipy.fft

WEIGHT = (1.0, 1.0, 2.0)  # the shear component counts twice in the strain energy


def fibers(N, count=30, radius=0.06, seed=0):
    """phase (N, N): 1 inside random non-overlapping periodic circular fibers, 0 elsewhere"""
    rng = np.random.default_rng(seed)
    centers = []
    while len(centers) < count:
        c = rng.random(2)
        d = [np.abs((c - o + 0.5) % 1 - 0.5) for o in centers]
        if all(np.hypot(*v) > 2.1 * radius for v in d):
            centers.append(c)
    x = (np.arange(N) + 0.5) / N
    phase = np.zeros((N, N))
    for c in centers:
        dx = (x[:, None] - c[0] + 0.5) % 1 - 0.5
        dy = (x[None, :] - c[1] + 0.5) % 1 - 0.5
        phase[dx**2 + dy**2 < radius**2] = 1
    return phase


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def gamma(txx, tyy, txy, kx, ky, lam0, mu0):
    """Green operator of the isotropic reference medium (lam0, mu0) applied to tau in Fourier
    space: the strain of the periodic displacement that the polarization tau induces"""
    k2 = kx * kx + ky * ky
    k2 = k2 + (k2 == 0)  # the mean is set elsewhere
    bx, by = txx * kx + txy * ky, txy * kx + tyy * ky  # tau k
    kb = (kx * bx + ky * by) * (lam0 + mu0) / ((lam0 + 2 * mu0) * k2)
    wx, wy = (bx - kx * kb) / (mu0 * k2), (by - ky * kb) / (mu0 * k2)
    keep = (kx * kx + ky * ky > 0) * (-kx < ky.size - 1) * (ky < ky.size - 1)  # mean and Nyquist zero
    return keep * kx * wx, keep * ky * wy, keep * 0.5 * (kx * wy + ky * wx)


def material(exx, eyy, exy, lam, mu):
    tr = lam * (exx + eyy)
    return tr + 2 * mu * exx, tr + 2 * mu * eyy, 2 * mu * exy


class Homogenization:
    def __init__(self, lam, mu, xp=np, dtype=np.float64):
        self.xp, self.dtype = xp, dtype
        self.fft = cupyx.scipy.fft if xp is cp else scipy.fft
        self.lam, self.mu = xp.asarray(lam, dtype), xp.asarray(mu, dtype)
        N = lam.shape[0]
        self.kx = xp.asarray(np.fft.fftfreq(N, 1 / N)[:, None], dtype)
        self.ky = xp.asarray(np.fft.rfftfreq(N, 1 / N)[None, :], dtype)
        self.E = xp.asarray(np.eye(3) / WEIGHT, dtype)[:, :, None, None]  # unit Voigt strains
        self.weight = xp.asarray(WEIGHT, dtype)[None, :, None, None]

    def stress(self, eps):
        return self.xp.stack(material(eps[:, 0], eps[:, 1], eps[:, 2], self.lam, self.mu), axis=1)

    def project(self, tau, lam0=0.0, mu0=0.5):
        """Gamma tau; the default reference (lam0, mu0) = (0, 1/2) is the compatibility projection"""
        t = self.fft.rfft2(tau, workers=-1) if self.xp is np else self.fft.rfft2(tau)
        g = self.xp.stack(gamma(t[:, 0], t[:, 1], t[:, 2], self.kx, self.ky, lam0, mu0), axis=1)
        N = tau.shape[-1]
        return self.fft.irfft2(g, s=(N, N), workers=-1) if self.xp is np else self.fft.irfft2(g, s=(N, N))

    def dot(self, a, b):  # one energy product per load case, shape (3, 1, 1, 1)
        return (self.weight * a * b).sum(axis=(1, 2, 3), keepdims=True, dtype=self.xp.float64)

    def basic(self, rtol=1e-6, maxiter=10000):
        """Moulinec-Suquet fixed point eps <- eps - Gamma0 sigma(eps) with the reference
        medium halfway between the extreme phases"""
        lam0 = 0.5 * (float(self.lam.min()) + float(self.lam.max()))
        mu0 = 0.5 * (float(self.mu.min()) + float(self.mu.max()))
        eps = self.xp.broadcast_to(self.E, (3, 3, *self.lam.shape)).copy()
        norm = self.dot(self.E, self.E) * self.lam.size
        for k in range(1, maxiter + 1):
            d = self.project(self.stress(eps), lam0, mu0)
            eps -= d
            if float((self.dot(d, d) / norm).max()) < rtol**2:
                break
        return eps, k

    def cg(self, rtol=1e-6, maxiter=10000):
        """conjugate gradients on the projected equilibrium G C (E + x) = 0 (Zeman et al.),
        three load cases at once"""
        r = -self.project(self.stress(self.xp.broadcast_to(self.E, (3, 3, *self.lam.shape))))
        x, p = self.xp.zeros_like(r), r.copy()
        rr = self.dot(r, r)
        tol = rtol**2 * float(rr.max())
        for k in range(1, maxiter + 1):
            Ap = self.project(self.stress(p))
            alpha = rr / self.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rr, rr_old = self.dot(r, r), rr
            if float(rr.max()) < tol:
                break
            p = r + rr / rr_old * p
        return x + self.E, k

    def effective(self, eps):
        """effective stiffness in Voigt notation: column j is the mean stress of load case j"""
        return self.stress(eps).mean(axis=(2, 3)).T


# ----------------------------------- fused GPU kernels -------------------------------
# one thread per pixel and load case reads and writes all three components; n = N * N
material_kernel = cp.ElementwiseKernel(
    "raw T eps, raw T lam, raw T mu, int64 n", "raw T sig",
    """
    const long p = i % n, a = (i / n) * 3 * n + p, b = a + n, c = b + n;
    const T tr = lam[p] * (eps[a] + eps[b]), m2 = 2 * mu[p];
    sig[a] = tr + m2 * eps[a]; sig[b] = tr + m2 * eps[b]; sig[c] = m2 * eps[c];
    """, "material_kernel")

# compatibility projection in place, the frequencies follow from the index; m = N * (N / 2 + 1)
gamma_kernel = cp.ElementwiseKernel(
    "int64 N, int64 m", "raw C t",
    """
    typedef C::value_type T;
    const long p = i % m, a = (i / m) * 3 * m + p, b = a + m, c = b + m;
    const long row = p / (N / 2 + 1);
    const T kx = row < N / 2 ? row : row - N, ky = p % (N / 2 + 1), k2 = kx * kx + ky * ky;
    if(k2 == 0 || row == N / 2 || ky == N / 2){ t[a] = t[b] = t[c] = 0; return; }  // mean and Nyquist
    const C bx = t[a] * kx + t[c] * ky, by = t[c] * kx + t[b] * ky;
    const C kb = (kx * bx + ky * by) / (2 * k2);
    const C wx = (bx - kx * kb) * (2 / k2), wy = (by - ky * kb) * (2 / k2);
    t[a] = kx * wx; t[b] = ky * wy; t[c] = (T)0.5 * (kx * wy + ky * wx);
    """, "gamma_kernel")

weighted_dot = cp.ReductionKernel(
    "T a, T b, T w", "float64 y", "w * (double)a * b", "a + b", "y = a", "0", "weighted_dot")

xr_update = cp.ElementwiseKernel(  # alpha per load case, n3 = 3 N^2 entries each
    "T p, T Ap, raw T alpha, int32 n3", "T x, T r",
    "const T a = alpha[(int)i / n3]; x += a * p; r -= a * Ap", "xr_update")

p_update = cp.ElementwiseKernel(
    "T r, raw T beta, int32 n3", "T p", "p = r + beta[(int)i / n3] * p", "p_update")


class FusedHomogenization(Homogenization):
    """the conjugate gradients of `Homogenization` with fused kernels, GPU only"""

    def __init__(self, lam, mu, dtype=cp.float32):
        super().__init__(lam, mu, cp, dtype)
        self.n = lam.size
        # the dot products reduce over 3 x 256 chunks per load case, so that many blocks work
        self.weight = cp.repeat(cp.asarray(WEIGHT, dtype), 256)[None, :, None]

    def stress(self, eps, sig=None):
        sig = cp.empty_like(eps) if sig is None else sig
        material_kernel(eps, self.lam, self.mu, self.n, sig, size=3 * self.n)
        return sig

    def project(self, tau):
        N = tau.shape[-1]
        t = cupyx.scipy.fft.rfft2(tau, overwrite_x=True)
        gamma_kernel(N, N * (N // 2 + 1), t, size=3 * N * (N // 2 + 1))
        return cupyx.scipy.fft.irfft2(t, s=(N, N), overwrite_x=True)

    def dot(self, a, b):
        chunks = weighted_dot(a.reshape(3, 768, -1), b.reshape(3, 768, -1), self.weight, axis=2)
        return chunks.sum(axis=1)

    def cg(self, rtol=1e-4, maxiter=10000):
        n3 = np.int32(3 * self.n)
        r = self.project(self.stress(cp.ascontiguousarray(cp.broadcast_to(self.E, (3, 3, *self.lam.shape)))))
        r *= -1
        x, p, sig = cp.zeros_like(r), r.copy(), cp.empty_like(r)
        rr = self.dot(r, r)
        tol = rtol**2 * rr.max().get()
        for k in range(1, maxiter + 1):
            Ap = self.project(self.stress(p, sig))
            xr_update(p, Ap, (rr / self.dot(p, Ap)).astype(self.dtype), n3, x, r)
            rr, rr_old = self.dot(r, r), rr
            if rr.max().get() < tol:
                break
            p_update(r, (rr / rr_old).astype(self.dtype), n3, p)
        return x + self.E, k
