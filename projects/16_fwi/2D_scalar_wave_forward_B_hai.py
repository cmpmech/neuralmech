import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import math
import cmasher as cmr
from solvers.scalar_wave import (setup_simulation, setup_source, simulate2D)
from numeric import sineburst

# -------------------------- problem definition --------------------------
# implementation
threads_j, threads_i = 128, 4
precompiled = False # True

# load material from CT scan
indicator = np.ascontiguousarray(np.load('../../data/B_Hai_1.npy').T)
indicator[:50,:50] = 1.
indicator[:5,:] = 1.
indicator[-6:,:] = 1.
indicator[:,:5] = 1.
indicator[:,-6:] = 1.
Nx, Ny = indicator.shape[0], indicator.shape[1]
print(Nx, Ny)
min_indicator = 1e-3
indicator[indicator == 0] = min_indicator

# physics
Lx, Ly = 0.04, 0.04 * Ny / Nx
wavespeed = 6000.
density = 2700.
amplitude = 1e8
cycles = 1
T = 8e-5

# discretization
dx, dy = Lx / (Nx - 3), Ly / (Ny - 3)
cfl_safety_factor = 0.95
dt = cfl_safety_factor * min(dx, dy) / wavespeed / math.sqrt(2)
frequency = wavespeed / 10. / dx
N = math.ceil(T / dt)

simulation = setup_simulation((Nx, Ny), (dx, dy), N, dt,
                              wavespeed, density, (threads_i, threads_j))

# source
x = np.linspace(-dx, Lx+dx, Nx)
y = np.linspace(-dy, Ly+dy, Ny)
x, y = np.meshgrid(x, y, indexing='ij')

sourceIndices = np.vstack(np.where(np.isclose(x, 0.) & np.isclose(y, 0.)))
sourceIndices = cp.asarray(sourceIndices, dtype=cp.int32)

t = np.linspace(0, (N - 1) * dt, N)
signal = sineburst(t, amplitude, frequency, cycles) / dx / dy # for the dirac
signal = cp.asarray(np.repeat(np.expand_dims(signal, axis=1), len(sourceIndices[0]), axis=1), dtype=cp.float32)
source = setup_source(sourceIndices, signal)

# material
indicator_padded = cp.ones(simulation.Nx_padded, dtype=cp.float32)
indicator_padded[:Nx, :Ny] = cp.asarray(indicator, dtype=cp.float32)

# -------------------------------- solve ---------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate2D(simulation, source, indicator_padded, precompiled=precompiled)
cp.cuda.Stream.null.synchronize()
toc = time.time()
elapsed_time = toc - tic
print(f'elapsed time: {elapsed_time:.3f} s')
print(f'elapsed time per timestep: {elapsed_time / N * 1e3:.4f} ms')
dofs = ((Nx - 2) * (Ny - 2))
print(f'elapsed time per timestep: {dofs / (elapsed_time / N)/1e9:.2f} billion dofs/s')

# --------------------------- post-processing ----------------------------
u_ = np.ma.masked_where(indicator == min_indicator, u.get())
indicator_ = np.ma.masked_where(indicator != min_indicator, indicator)
scale = cp.max(cp.abs(u)) * 3e-2
fig, ax = plt.subplots(figsize=(Nx / 150,Ny / 150), dpi=150)
cb = ax.pcolormesh(x, y, u_, cmap=cmr.fusion,
                   vmin=-scale, vmax=scale)
ax.pcolormesh(x, y, indicator_ * 0 + 0.7, cmap='Greys_r', vmin=0, vmax=1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig('../../results/CTwaves2D.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('../../results/CTwaves2D.png', bbox_inches='tight', pad_inches=0)
plt.show()