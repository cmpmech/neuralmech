from pathlib import Path

import numpy as np
import cupy as cp
from PIL import Image
import time
import math
import cmasher as cmr
from solvers.scalar_wave import (setup_simulation, setup_source, simulate2D)
from numeric import sineburst

BASE_DIR = Path(__file__).parent

plot_every = 2

# -------------------------- problem definition --------------------------
# implementation
threads_j, threads_i = 128, 4
precompiled = False # True

# load material from CT scan
indicator = np.ascontiguousarray(np.load(BASE_DIR / '../../data/B_Hai_1.npy').T)
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
T = 1e-5

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
for n in range(0, N, plot_every):
    simulation.N = n
    u = simulate2D(simulation, source, indicator_padded, precompiled=precompiled)

    u_arr = u.get()
    scale = 5e-6

    # normalize wave field to [0, 1] and apply colormap
    data_norm = np.clip((u_arr + scale) / (2 * scale), 0, 1)
    rgb = (cmr.fusion(data_norm)[:, :, :3] * 255).astype(np.uint8)

    # overlay void regions with grey (Greys_r at 0.7 → RGB 178,178,178)
    void_mask = (indicator == min_indicator)
    rgb[void_mask] = 178

    img = Image.fromarray(rgb, mode='RGB')
    img.save(f'../../results/animations/animation_frames/bhai_waves/frame_{n // plot_every}.jpg',
             quality=85)