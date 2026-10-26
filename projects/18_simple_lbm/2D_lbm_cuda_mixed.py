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
f = lbm.equilibrium(ux, uy, DTYPE)
f_new = cp.empty_like(f)

# --------------------------------------- setup ---------------------------------------
module = cp.RawModule(code=(BASE_DIR / "lbm.cu").read_text(), options=COMPILER_OPTIONS)
fused = module.get_function("fused_step")
blocks = ((NY + THREADS - 1) // THREADS, NX)


def lbm_step(f, f_new):
    fused(
        blocks,
        (THREADS,),
        (f, f_new, omega_plus, omega_minus, np.int32(NX), np.int32(NY)),
    )
    return f_new, f


# --------------------------------------- solve ---------------------------------------
lbm_step(f.copy(), cp.empty_like(f))  # warm-up: the first launch compiles the kernel
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in range(STEPS):
    f, f_new = lbm_step(f, f_new)
cp.cuda.Stream.null.synchronize()
toc = time.time()

print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / STEPS * 1e3:.4f} ms/step)")
print(f"{NX * NY / ((toc - tic) / STEPS) / 1e9:.2f} billion node updates/s")

# --------------------------------- single precision ----------------------------------
module_32 = cp.RawModule(
    code=(BASE_DIR / "lbm.cu").read_text(), options=("--use_fast_math",)
)
fused_32 = module_32.get_function("fused_step")
f_32 = lbm.equilibrium(ux, uy)
f_32_new = cp.empty_like(f_32)
for t in range(STEPS):
    fused_32(
        blocks,
        (THREADS,),
        (f_32, f_32_new, omega_plus, omega_minus, np.int32(NX), np.int32(NY)),
    )
    f_32, f_32_new = f_32_new, f_32

u, u_32 = cp.stack(lbm.velocity(f)), cp.stack(lbm.velocity(f_32))
deviation = float(cp.linalg.norm(u - u_32) / cp.linalg.norm(u_32))
print(f"relative deviation from single precision {deviation:.2e}")

# ----------------------------------- postprocessing ----------------------------------
vorticity = lbm.vorticity(*lbm.velocity(f))[::8, ::8].get()
scale = 0.5 * abs(vorticity).max()
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
ax.imshow(vorticity.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
