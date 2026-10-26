import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt

import lbm

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float32

# physics
VELOCITY = 0.05  # rms velocity, Mach number 0.09
WAVELENGTH = 200  # nodes, of the most energetic vortices at the start
VISCOSITY = 0.008

# discretization
NX, NY = 8192, 4096
STEPS = 1200

# ------------------------------------ prepare data -----------------------------------
omega_plus, omega_minus = lbm.relaxation_rates(VISCOSITY)
ux, uy = lbm.random_vortices(NX, NY, WAVELENGTH, VELOCITY)
f = lbm.equilibrium(ux, uy, DTYPE)

cx = cp.asarray(lbm.C[:, 0], DTYPE)
cy = cp.asarray(lbm.C[:, 1], DTYPE)
w = cp.asarray(lbm.W, DTYPE)[:, None, None]


def lbm_step(f):
    g = cp.stack([cp.roll(f[i], tuple(lbm.C[i]), axis=(0, 1)) for i in range(9)])
    drho = g.sum(0)
    rho = 1 + drho
    ux = cp.tensordot(cx, g, axes=1) / rho
    uy = cp.tensordot(cy, g, axes=1) / rho
    cu = 3 * (cx[:, None, None] * ux + cy[:, None, None] * uy)
    plus = w * (drho + rho * (0.5 * cu**2 - 1.5 * (ux**2 + uy**2)))
    minus = w * rho * cu
    g_opp = g[lbm.OPP]
    sym = omega_plus * (0.5 * (g + g_opp) - plus)
    asym = omega_minus * (0.5 * (g - g_opp) - minus)
    return g - sym - asym


# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in range(STEPS):
    f = lbm_step(f)
cp.cuda.Stream.null.synchronize()
toc = time.time()

print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / STEPS * 1e3:.4f} ms/step)")
print(f"{NX * NY / ((toc - tic) / STEPS) / 1e9:.2f} billion node updates/s")

# ----------------------------------- postprocessing ----------------------------------
vorticity = lbm.vorticity(*lbm.velocity(f))[::8, ::8].get()
scale = 0.5 * abs(vorticity).max()
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
ax.imshow(vorticity.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
