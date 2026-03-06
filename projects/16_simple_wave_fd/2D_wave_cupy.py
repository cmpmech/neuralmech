import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import cmasher as cmr

# -------------------------- problem definition --------------------------
# implementation
dtype = cp.float32

# physics
Lx, Ly = 1., 2.
wavespeed = 1.
T = 1.

# discretization
Nx, Ny = 1600, 3200
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
# cupy
def fd_step(u0, u1, u2):
    laplacian_x = (u1[:-2,1:-1] - 2 * u1[1:-1,1:-1] + u1[2:,1:-1]) / dx**2
    laplacian_y = (u1[1:-1,:-2] - 2 * u1[1:-1,1:-1] + u1[1:-1,2:]) / dy**2
    u2[1:-1,1:-1] = (-u0[1:-1,1:-1] + 2 * u1[1:-1,1:-1] +
                     dt**2 * wavespeed**2 * (laplacian_x + laplacian_y))
    return u2

def bc_step(u):
    u[0,:] = 0
    u[-1,:] = 0
    u[:,0] = 0
    u[:,-1] = 0
    return u

# -------------------------------- solve ---------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
for t in tqdm(range(N)):
    u0 = fd_step(u0, u1, u0)
    u1, u0 = u0, u1
    # u1 = bc_step(u1) # not necessary
toc = time.time()
elapsed_time = toc - tic
print(f'elapsed time: {elapsed_time:.3f} s')
print(f'elapsed time per timestep: {elapsed_time / N * 1e3:.3f} ms')
dofs = ((Nx - 2) * (Ny - 2))
print(f'elapsed time per timestep: {dofs / (elapsed_time / N)/1e9:.2f} billion dofs/s')

# ---------------------------- postprocessing ----------------------------
scale = cp.max(cp.abs(u1))
fig, ax = plt.subplots()
cb = ax.pcolormesh(x, y, u1.get(), cmap=cmr.fusion,
                   vmin=-scale, vmax=scale)
ax.set_aspect('equal')
fig.colorbar(cb)
plt.show()