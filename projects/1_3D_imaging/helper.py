import numpy as np
import torch
import torch.nn as nn


# ------------------------------- forward operator ------------------------------------
def A_op(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masking forward operator: keep the measured pixels, zero the rest."""
    return mask * x


def At_op(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Adjoint of the masking operator (A^T = A for a 0/1 mask)."""
    return mask * y


def cg(A_fn, rhs: torch.Tensor, n_iter: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """Conjugate-gradient solve of A_fn(x) = rhs from x0 = 0, batched and differentiable.

    A_fn is any callable applying a symmetric positive-definite operator; kept
    torch-native (rather than scipy) so it runs on the GPU, handles a batch at
    once, and stays in the autograd graph for backprop through unrolled solvers.
    """
    x = torch.zeros_like(rhs)
    r = rhs.clone()  # rhs - A_fn(0) = rhs
    p = r.clone()
    rs_old = (r * r).sum()

    for _ in range(n_iter):
        Ap = A_fn(p)
        alpha = rs_old / ((p * Ap).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new

    return x


def cg_solve(
    mask: torch.Tensor, rhs: torch.Tensor, lam: float, n_iter: int = 10
) -> torch.Tensor:
    """CG solve of the MoDL normal equations (A^T A + lambda I) x = rhs."""

    def Lx(v):
        return At_op(A_op(v, mask), mask) + lam * v

    return cg(Lx, rhs, n_iter)


# ------------------------------------ MoDL model -------------------------------------
class MoDL(nn.Module):
    def __init__(self, denoiser: nn.Module, K: int, lam: float, cg_iter: int):
        super().__init__()
        self.K = K
        self.lam = lam
        self.cg_iter = cg_iter
        self.D_w = denoiser

    def forward(self, b, mask):  # K unrolled iterations
        x = b.clone()
        for _ in range(self.K):
            z = self.D_w(x)  # D_w is shared across all K iterations
            rhs = At_op(b, mask) + self.lam * z
            x = cg_solve(mask, rhs, self.lam, self.cg_iter)
        return x


# ---------------------------------- ultrasound (DAS) ---------------------------------
def born_forward(
    emitter_pos: np.ndarray,
    sensor_positions: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    c: float,
    pulse,
) -> np.ndarray:
    """Born-approximation A-scans for one emitter: direct arrival + defect reflections.

    `pulse` is a callable mapping time offsets to excitation amplitude. Returns an
    array of shape (n_sensors, len(t)).
    """
    signals = np.zeros((len(sensor_positions), len(t)))
    for j, sensor_pos in enumerate(sensor_positions):
        tof_direct = np.linalg.norm(sensor_pos - emitter_pos) / c
        signals[j] += pulse(t - tof_direct)

    for ix in range(z.shape[0]):
        for iy in range(z.shape[1]):
            if z[ix, iy] == 0.0:
                continue
            cell = np.array([x[ix], y[iy]])
            d_emit = np.linalg.norm(cell - emitter_pos)
            for j, sensor_pos in enumerate(sensor_positions):
                tof = (d_emit + np.linalg.norm(cell - sensor_pos)) / c
                signals[j] += z[ix, iy] * pulse(t - tof)
    return signals


def das_backproject(
    emitter_pos: np.ndarray,
    sensor_positions: np.ndarray,
    signals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    c: float,
) -> np.ndarray:
    """Delay-and-sum backprojection of one emitter's A-scans onto the (NX, NY) grid."""
    das = np.zeros((len(x), len(y)))
    for ix in range(len(x)):
        for iy in range(len(y)):
            cell = np.array([x[ix], y[iy]])
            d_emit = np.linalg.norm(cell - emitter_pos)
            for j, sensor_pos in enumerate(sensor_positions):
                tof = (d_emit + np.linalg.norm(cell - sensor_pos)) / c
                das[ix, iy] += np.interp(tof, t, signals[j])
    return das
