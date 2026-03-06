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
T = 1.

# discretization
Nx, Ny = 1600, 3200
# Nx, Ny = 1600 - 1, 3200 - 1
# Nx, Ny = 1600 + 32, 3200 + 32 # faster than -1
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
cfl_safety_factor = 0.95
dt = cfl_safety_factor * min(dx, dy) / wavespeed / math.sqrt(2)
N = math.ceil(T / dt)

# initial condition (square)
U = cp.zeros((2, Nx, Ny), dtype=dtype)
u0 = U[0]
u1 = U[1]
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
# cuda V2
if precompiled:
    compiled_kernels = cp.RawModule(path='step2D_wave.ptx')
    fd_kernel = compiled_kernels.get_function('fd_kernelV2')
else:
    fd_kernel = cp.RawKernel(open('step2D_wave.cu').read(),
                             'fd_kernelV2',
                             compiler_options)

blocks_j = (Ny + threads_j - 1) // threads_j # cols (Ny is num cols)
blocks_i = (Nx + threads_i - 1) // threads_i # rows (Nx is num rows)

def fd_step(u0, u1, u2):
    fd_kernel((blocks_j, blocks_i),(threads_j, threads_i),
              (u0, u1, u2, dtype(wavespeed), dtype(dt),
                    dtype(dx), dtype(dy), Nx, Ny))
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
    cb = ax.pcolormesh(x, y, u1.get(), cmap=cmr.fusion,
                       vmin=-scale, vmax=scale)
    ax.set_aspect('equal')
    fig.colorbar(cb)
    plt.show()