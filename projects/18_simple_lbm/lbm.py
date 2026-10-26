"""D2Q9 lattice tables, initial state and diagnostics shared by the drivers.

Lattice units (spacing and time step 1). Arrays are (Nx, Ny) with y the fast index,
populations are stored as their deviation g = f - w from rest, and the grid is periodic
in both directions.
"""

import cupy as cp
import numpy as np

C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def relaxation_rates(viscosity, magic=0.25):
    """the two rates of the two-relaxation-time collision; magic 1/4 for stability."""
    tau = 3 * viscosity + 0.5
    return np.float32(1 / tau), np.float32(1 / (magic / (tau - 0.5) + 0.5))


def equilibrium(ux, uy, dtype=cp.float32):
    """deviation f_eq - w at density 1, of shape (9, *ux.shape)."""
    g = cp.empty((9, *ux.shape), dtype)
    for i in range(9):
        cu = 3 * (C[i, 0] * ux + C[i, 1] * uy)
        g[i] = W[i] * (cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))
    return g


def velocity(g):
    """(ux, uy) in float32 from the deviations g of any storage type."""
    rho = cp.ones(g.shape[1:], cp.float32)
    jx, jy = cp.zeros_like(rho), cp.zeros_like(rho)
    for i in range(9):
        gi = g[i].astype(cp.float32)
        rho += gi
        jx += C[i, 0] * gi
        jy += C[i, 1] * gi
    return jx / rho, jy / rho


def vorticity(ux, uy):
    """dv/dx - du/dy by central differences on the periodic grid."""
    return (
        cp.roll(uy, -1, 0) - cp.roll(uy, 1, 0) - cp.roll(ux, -1, 1) + cp.roll(ux, 1, 1)
    ) / 2


def stream(g):
    """pull streaming g_i(x) <- g_i(x - c_i) without collision (for the checks)."""
    return cp.stack([cp.roll(g[i], tuple(C[i]), axis=(0, 1)) for i in range(9)])


def random_vortices(Nx, Ny, wavelength, rms, seed=0):
    """divergence-free random field (ux, uy) with the spectrum k^4 exp(-2 (k / k0)^2).

    The most energetic vortices have the wavelength 1 / k0 in nodes, and the field is
    scaled to the rms velocity `rms`. Built with numpy, as it is computed only once.
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(Nx)[:, None]  # periods per node
    ky = np.fft.rfftfreq(Ny)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1
    # stream function amplitude, since E(k) ~ k^3 |psi|^2
    amplitude = np.sqrt(k) * np.exp(-((k * wavelength) ** 2))
    amplitude[0, 0] = 0
    psi = (amplitude * np.exp(2j * np.pi * rng.random(k.shape))).astype(np.complex64)
    ux = np.fft.irfft2(2j * np.pi * ky * psi, (Nx, Ny)).astype(np.float32)
    uy = np.fft.irfft2(-2j * np.pi * kx * psi, (Nx, Ny)).astype(np.float32)
    scale = rms / np.sqrt((ux**2 + uy**2).mean())
    return cp.asarray(scale * ux), cp.asarray(scale * uy)
