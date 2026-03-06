import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import cmasher as cmr

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

# ------------------------------- padding --------------------------------
Nx_padded = Nx
Ny_padded = ((Ny + 32 - 1) // 32) * 32

# initial condition (square)
U = cp.zeros((2, Nx_padded, Ny_padded), dtype=cp.float16)
u0 = U[0]
u1 = U[1]
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
x, y = np.meshgrid(x, y, indexing='ij')

# gaussian
x0, y0 = 0.5, 0.5  # center
sigma = 0.02       # width
u0[:Nx,:Ny] = cp.asarray(np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)))
u1[:] = u0[:]

# homogeneous Dirichlet boundary conditions

# --------------------------- simulation setup ---------------------------
# cuda V5
if precompiled:
    compiled_kernels = cp.RawModule(path='step2D_wave.ptx')
    fd_kernel = compiled_kernels.get_function('fd_kernelV5')
else:
    fd_kernel = cp.RawKernel(open('step2D_wave.cu').read(),
                             'fd_kernelV5',
                             compiler_options)

blocks_j = (Ny_padded + threads_j - 1) // threads_j # cols (Ny is num cols)
blocks_i = (Nx_padded + threads_i - 1) // threads_i # rows (Nx is num rows)

def fd_step(u0, u1, u2):
    fd_kernel((blocks_j, blocks_i),(threads_j, threads_i),
              (u0, u1, u2, dtype(wavespeed), dtype(dt),
                    dtype(dx), dtype(dy), Nx, Ny, Ny_padded))
    return u2

# -------------------------------- solve ---------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in tqdm(range(N)):
    u0 = fd_step(u0, u1, u0)
    u1, u0 = u0, u1
cp.cuda.Stream.null.synchronize()
toc = time.time()
elapsed_time = toc - tic
print(f'elapsed time: {elapsed_time:.3f} s')
print(f'elapsed time per timestep: {elapsed_time / N * 1e3:.4f} ms')
dofs = ((Nx - 2) * (Ny - 2))
print(f'elapsed time per timestep: {dofs / (elapsed_time / N)/1e9:.2f} billion dofs/s')

# ---------------------------- postprocessing ----------------------------
if Nx <= 1600:
    scale = cp.max(cp.abs(u1))
    fig, ax = plt.subplots()
    cb = ax.pcolormesh(x, y, u1[:Nx,:Ny].get(), cmap=cmr.fusion,
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
# plt.savefig(f'../../results/wave_mixed_{T}.pdf', bbox_inches='tight', pad_inches=0)
# plt.close()