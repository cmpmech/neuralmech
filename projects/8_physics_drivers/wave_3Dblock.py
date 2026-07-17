import math
import time

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from solvers.wave import scalar_simulation, setup_source, simulate

# -------------------------------------- settings -------------------------------------
THREADS = (4, 4, 64)

# physics
LENGTH_X = 0.3  # m, 30 cm block length
LENGTH_YZ = 0.15  # m, 15x15 cm cross section
HOLE_DIAMETER = 0.03  # m, 4 cm duct through the entire depth (z), centered mid-length
WAVESPEED = 4000.0  # m/s,
DENSITY = 2400.0  # kg/m^3, normal-weight concrete
FREQUENCY = 150e3  # Hz
POINTS_PER_WAVELENGTH = 12
TRAVERSALS = 1  # full back-and-forth trips along LENGTH_X to simulate
AMPLITUDE = 1e8
CYCLES = 5
MIN_INDICATOR = 1e-3
CFL = 0.95

# --------------------------------------- setup ---------------------------------------
dx_target = WAVESPEED / FREQUENCY / POINTS_PER_WAVELENGTH
LENGTHS = (LENGTH_X, LENGTH_YZ, LENGTH_YZ)
Nx = tuple(round(L / dx_target) + 3 for L in LENGTHS)
dx = tuple(L / (n - 3) for L, n in zip(LENGTHS, Nx))
dt = CFL * min(dx) / WAVESPEED / math.sqrt(3)
T = TRAVERSALS * 2 * LENGTH_X / WAVESPEED
N = math.ceil(T / dt)
dofs = math.prod(n - 2 for n in Nx)
print(f"mesh size {Nx}, {dofs:.2e} dofs")

sim = scalar_simulation(Nx, dx, N, dt, THREADS, wavespeed=WAVESPEED, density=DENSITY)

x = np.linspace(-dx[0], LENGTH_X + dx[0], Nx[0])
y = np.linspace(-dx[1], LENGTH_YZ + dx[1], Nx[1])
x, y = np.meshgrid(x, y, indexing="ij")
hole = (x - LENGTH_X / 2) ** 2 + (y - LENGTH_YZ / 2) ** 2 < (HOLE_DIAMETER / 2) ** 2

indicator = np.ones(Nx, dtype=np.float32)
indicator[hole] = MIN_INDICATOR

indicator_padded = cp.ones(sim.Nx_padded, dtype=sim.dtype)
indicator_padded[: Nx[0], : Nx[1], : Nx[2]] = cp.asarray(indicator, dtype=sim.dtype)

source_pos = cp.array([[1], [Nx[1] // 2], [Nx[2] // 2]], dtype=cp.int32)


# --------------------------------------- helper --------------------------------------
def sineburst(t, amplitude, frequency, cycles):
    mask = (t > 0) & (t <= cycles / frequency)
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * frequency * t)
        * np.sin(np.pi * frequency * t / cycles) ** 2
    )


t = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)
source = setup_source(source_pos, signal)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator_padded)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s ({(toc - tic) / N * 1e3:.4f} ms/step)")

# ----------------------------------- postprocessing ----------------------------------
u_np = u.get()
z_mid = Nx[2] // 2
field = u_np[:, :, z_mid]
indicator_slice = indicator[:, :, z_mid]

field_masked = np.ma.masked_where(indicator_slice == MIN_INDICATOR, field)
void_masked = np.ma.masked_where(indicator_slice != MIN_INDICATOR, indicator_slice)
scale = float(np.max(np.abs(field_masked)))  # void excitation dwarfs the solid field

fig, ax = plt.subplots(figsize=(8, 4))
ax.pcolormesh(x, y, field_masked, cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.pcolormesh(x, y, void_masked * 0 + 0.7, cmap="Greys_r", vmin=0, vmax=1)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
