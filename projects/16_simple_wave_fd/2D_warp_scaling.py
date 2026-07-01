import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import pandas as pd

# -------------------------- problem definition --------------------------
# implementation
wp.init()
# wp.config.mode = "release"
# wp.config.verify_cuda = False
# if hasattr(wp.config, 'nvrtc_options'):
#     wp.config.nvrtc_options.append("--use_fast_math")

dtype = float # wp.float64

# physics
Lx, Ly = 1., 2.
wavespeed = 1.
T = 1.

# --------------------------- parametric study ---------------------------
dofs = []
timings = []
nlist = np.logspace(0.5, 4.5, 50).astype(np.int32)
for n in nlist:
    # discretization
    Nx, Ny = n, n
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
    cfl_safety_factor = 0.95
    dt = cfl_safety_factor * min(dx, dy) / wavespeed / math.sqrt(2)
    N = 2000

    # initial condition (square)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    x, y = np.meshgrid(x, y, indexing='ij')

    # gaussian
    x0, y0 = 0.5, 0.5  # center
    sigma = 0.02  # width
    u_init = np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)).astype(
        np.float32)


    # warp arrays on GPU
    u0 = wp.array(u_init, dtype=dtype, device="cuda")
    u1 = wp.array(u_init, dtype=dtype, device="cuda")

    # --------------------------- simulation setup ---------------------------
    # warp kernel
    @wp.kernel(enable_backward=False)
    def wave_step_kernel(
            u_prev: wp.array2d(dtype=dtype),
            u_curr: wp.array2d(dtype=dtype),
            u_next: wp.array2d(dtype=dtype), # u_prev in kernel (reused)
            c_sq: float,
            dt_sq: float,
            inv_dx_sq: float,
            inv_dy_sq: float
    ):
        # thread idx
        i, j = wp.tid()
        # grid dimensions
        nx, ny = u_curr.shape[0], u_curr.shape[1]

        if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
            center = u_curr[i, j]

            laplacian_x = (u_curr[i - 1, j] - 2.0 * center + u_curr[
                           i + 1, j]) * inv_dx_sq
            laplacian_y = (u_curr[i, j - 1] - 2.0 * center + u_curr[
                i, j + 1]) * inv_dy_sq

            # (u_next here points to the memory of u_prev from the python loop)
            u_next[i, j] = 2.0 * center - u_prev[i, j] + (c_sq * dt_sq) * (
                           laplacian_x + laplacian_y)

    # precomputed constants for kernel
    c_sq = wavespeed ** 2
    dt_sq = dt ** 2
    inv_dx_sq = 1.0 / (dx ** 2)
    inv_dy_sq = 1.0 / (dy ** 2)

    # -------------------------------- solve ---------------------------------
    wp.synchronize()
    tic = time.time()
    for t in tqdm(range(N)):
        wp.launch(
            kernel=wave_step_kernel,
            dim=(Nx, Ny),
            inputs=[u0, u1, u0, c_sq, dt_sq, inv_dx_sq, inv_dy_sq],
            device="cuda",
            block_dim=128
        )
        u1, u0 = u0, u1
    wp.synchronize()
    toc = time.time()
    elapsed_time = toc - tic
    timings.append(elapsed_time / N)
    dofs.append((Nx - 2) * (Ny - 2))

    # -------------------------------- export --------------------------------
    df = pd.DataFrame({'x': dofs,
                       'y': timings})
    df.to_csv(f'../../results/data/warp_scaling.csv', sep=' ', index=False)

# ---------------------------- postprocessing ----------------------------
print(np.max(np.array(dofs) / np.array(timings))/1e9)
print(max(dofs)/1e6)

fig, ax = plt.subplots()
ax.plot(dofs, np.array(dofs) / np.array(timings), 'k')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
plt.show()