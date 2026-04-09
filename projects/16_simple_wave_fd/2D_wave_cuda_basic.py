from pathlib import Path

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import cmasher as cmr

BASE_DIR = Path(__file__).parent

# -------------------------- problem definition --------------------------
# implementation
threads_j, threads_i = 128, 4
dtype = cp.float32
# compiler_options = ()
compiler_options = ('--use_fast_math',)
precompiled = False

# physics
Lx, Ly = 1., 2.
wavespeed = 1.
T = 1

# discretization
Nx, Ny = 1600, 3200
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
cfl_safety_factor = 0.95
dt = cfl_safety_factor * min(dx, dy) / wavespeed / math.sqrt(2)
N = max(math.ceil(T / dt), 1)

# initial condition (square)
U = cp.zeros((2, Nx, Ny), dtype=dtype)
u0 = U[0]
u1 = U[1]
# u0 = cp.zeros((Nx, Ny), dtype=dtype)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
x, y = np.meshgrid(x, y, indexing='ij')

# gaussian
x0, y0 = 0.5, 0.5  # center
sigma = 0.02       # width
u0[:] = cp.asarray(np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)))
u1[:] = u0[:]

# homogeneous Dirichlet boundary conditions

# --------------------------- simulation setup ---------------------------
# cuda V1
if precompiled:
    compiled_kernels = cp.RawModule(path=str(BASE_DIR / 'step2D_wave.ptx'))
    fd_kernel = compiled_kernels.get_function('fd_kernelV1')
else:
    fd_kernel = cp.RawKernel(open(BASE_DIR / 'step2D_wave.cu').read(),
                             'fd_kernelV1',
                             compiler_options)


blocks_j = (Ny + threads_j - 1) // threads_j # cols (Ny is num cols)
blocks_i = (Nx + threads_i - 1) // threads_i # rows (Nx is num cols)
def fd_step(u0, u1, u2):
    fd_kernel((blocks_i, blocks_j),(threads_i, threads_j),
              (u0, u1, u2, dtype(wavespeed), dtype(dt),
                    dtype(dx), dtype(dy), Nx, Ny))
    return u2

# -------------------------------- solve ---------------------------------
tic = time.time()
cp.cuda.Stream.null.synchronize()
for t in tqdm(range(N)):
    u0 = fd_step(u0, u1, u0)
    u1, u0 = u0, u1
cp.cuda.Stream.null.synchronize()
toc = time.time()
elapsed_time = toc - tic
print(f'elapsed time: {elapsed_time:.3f} s')
print(f'elapsed time per timestep: {elapsed_time / N * 1e3:.3f} ms')
dofs = ((Nx - 2) * (Ny - 2))
print(f'elapsed time per timestep: {dofs / (elapsed_time / N)/1e9:.2f} billion dofs/s')

# ---------------------------- postprocessing ----------------------------
if Nx <= 1600:
    scale = cp.max(cp.abs(u1))
    fig, ax = plt.subplots()
    cb = ax.pcolormesh(x, y, u1.get(), cmap=cmr.fusion,
                       vmin=-scale, vmax=scale)
    ax.set_aspect('equal')
    fig.colorbar(cb)
    plt.show()

# -------------------------------- export --------------------------------
# fig, ax = plt.subplots(figsize=(3,6), dpi=100)
# scale = 0.1
# cb = ax.pcolormesh(x, y, u1.get(), cmap=cmr.fusion,
#                    vmin=-scale, vmax=scale)
# plt.gca().axes.get_yaxis().set_visible(False)
# plt.gca().axes.get_xaxis().set_visible(False)
# for spine in ax.spines.values():
#     spine.set_visible(False)
# plt.minorticks_off()
# ax.set_rasterized(True)
# fig.tight_layout(pad=0)
# plt.savefig(f'../../results/wave_full_{T}.pdf', bbox_inches='tight', pad_inches=0)
# plt.close()