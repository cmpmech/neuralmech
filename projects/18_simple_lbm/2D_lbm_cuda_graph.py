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
STEPS_PER_GRAPH = 20

# physics
VELOCITY = 0.05  # rms velocity, Mach number 0.09
WAVELENGTH = 128  # nodes, of the most energetic vortices at the start
VISCOSITY = 0.008

# discretization
NX, NY = 512, 256
STEPS = 20000

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
launches = (time.time() - tic) / STEPS

stream = cp.cuda.Stream(non_blocking=True)
with stream:
    stream.begin_capture()
    for t in range(
        STEPS_PER_GRAPH
    ):  # an even number, so f and f_new return to their roles
        f, f_new = lbm_step(f, f_new)
    graph = stream.end_capture()

    graph.launch(stream)
    stream.synchronize()
    tic = time.time()
    for t in range(STEPS // STEPS_PER_GRAPH):
        graph.launch(stream)
    stream.synchronize()
    toc = time.time()

print(
    f"single launches {launches * 1e6:.2f} us/step ({NX * NY / launches / 1e9:.2f} billion node updates/s)"
)
print(
    f"CUDA graph {(toc - tic) / STEPS * 1e6:.2f} us/step ({NX * NY / ((toc - tic) / STEPS) / 1e9:.2f} billion node updates/s)"
)

# ----------------------------------- postprocessing ----------------------------------
vorticity = lbm.vorticity(*lbm.velocity(f)).get()
scale = 0.5 * abs(vorticity).max()
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
ax.imshow(vorticity.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
