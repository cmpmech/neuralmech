from pathlib import Path

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import pandas as pd

BASE_DIR = Path(__file__).parent

for version in range(0, 2): # 0 is standard, 1 is mixed

    # -------------------------- problem definition --------------------------
    # implementation
    threads_j, threads_i = 128, 4
    dtype = cp.float32

    # physics
    Lx, Ly = 1., 1.
    wavespeed = 1.
    T = 1.

    # --------------------------- parametric study ---------------------------
    dofs = []
    timings = []
    nlist = np.logspace(0.5,4.5, 50).astype(np.int32)
    for n in nlist:

        # discretization
        Nx, Ny = n, n
        dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
        cfl_safety_factor = 0.95
        dt = cfl_safety_factor * min(dx, dy) / wavespeed / math.sqrt(2)
        N = 2000

        # ------------------------------- padding --------------------------------
        Nx_padded = Nx
        Ny_padded = ((Ny + 32 - 1) // 32) * 32

        # initial condition (square)
        if version == 0:
            U = cp.zeros((2, Nx_padded, Ny_padded), dtype=dtype)
        elif version == 1:
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
        blocks_j = (Ny_padded + threads_j - 1) // threads_j  # cols (Ny is num cols)
        blocks_i = (Nx_padded + threads_i - 1) // threads_i  # rows (Nx is num rows)
        if version == 0:
            compiled_kernels = cp.RawModule(path=str(BASE_DIR / 'step2D_wave.ptx'))
            fd_kernel = compiled_kernels.get_function('fd_kernelV3')
        elif version == 1:
            compiled_kernels = cp.RawModule(path=str(BASE_DIR / 'step2D_wave.ptx'))
            fd_kernel = compiled_kernels.get_function('fd_kernelV5')

        def fd_step(u0, u1, u2):
            fd_kernel((blocks_j, blocks_i), (threads_j, threads_i),
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
        timings.append(elapsed_time / N)
        dofs.append((Nx - 2) * (Ny - 2))

    # -------------------------------- export --------------------------------
    df = pd.DataFrame({'x': dofs,
                       'y': timings})
    df.to_csv(f'../../results/data/wave_scaling_{version}.csv', sep=' ', index=False)

    # --------------------------- post-processing ----------------------------
    print(np.max(np.array(dofs) / np.array(timings))/1e9)
    print(max(dofs)/1e6)

    fig, ax = plt.subplots()
    ax.plot(dofs, np.array(dofs) / np.array(timings), 'k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    plt.show()


