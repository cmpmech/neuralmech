import time
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import lbm

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# implementation
DTYPE = cp.float16
THREADS = 128
COMPILER_OPTIONS = ("--use_fast_math", "-Dstore=__half")

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
f = lbm.equilibrium(ux, uy, DTYPE)  # the only copy of the populations

# --------------------------------------- setup ---------------------------------------
module = cp.RawModule(code=(BASE_DIR / "lbm.cu").read_text(), options=COMPILER_OPTIONS)
aa_even = module.get_function("aa_even")
aa_odd = module.get_function("aa_odd")
fused = module.get_function("fused_step")
blocks = ((NY + THREADS - 1) // THREADS, NX)


def lbm_steps(f):  # two steps, in place
    aa_even(
        blocks, (THREADS,), (f, omega_plus, omega_minus, np.int32(NX), np.int32(NY))
    )
    aa_odd(blocks, (THREADS,), (f, omega_plus, omega_minus, np.int32(NX), np.int32(NY)))


# --------------------------------------- solve ---------------------------------------
lbm_steps(f.copy())  # warm-up: the first launch compiles the kernels
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in range(STEPS // 2):
    lbm_steps(f)
cp.cuda.Stream.null.synchronize()
toc = time.time()

print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / STEPS * 1e3:.4f} ms/step)")
print(f"{NX * NY / ((toc - tic) / STEPS) / 1e9:.2f} billion node updates/s")
print(
    f"{cp.cuda.Device().mem_info[1] / (9 * f.itemsize) / 1e9:.2f} billion nodes fit into the GPU"
)

# ------------------------------------ comparison -------------------------------------
# fused_step from the state after the first collision, one step behind, then streamed
g = lbm.equilibrium(ux, uy, DTYPE)
aa_even(blocks, (THREADS,), (g, omega_plus, omega_minus, np.int32(NX), np.int32(NY)))
g, g_new = g[lbm.OPP], cp.empty_like(g)
for t in range(STEPS - 1):
    fused(
        blocks,
        (THREADS,),
        (g, g_new, omega_plus, omega_minus, np.int32(NX), np.int32(NY)),
    )
    g, g_new = g_new, g
print(f"largest difference to fused_step {float(cp.abs(f - lbm.stream(g)).max()):.2e}")

# ----------------------------------- postprocessing ----------------------------------
vorticity = lbm.vorticity(*lbm.velocity(f))[::8, ::8].get()
scale = 0.5 * abs(vorticity).max()
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
ax.imshow(vorticity.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
