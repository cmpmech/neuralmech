import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from solvers.wave import scalar_simulation, setup_source
from solvers.wave_sensitivity import compute_sensitivity, compute_subtracted_kernels

# -------------------------------------- settings -------------------------------------
THREADS = (4, 4, 64)

# physics
LENGTH_X = 0.3  # m, 30 cm block length
LENGTH_YZ = 0.15  # m, 15x15 cm cross section
HOLE_DIAMETER = 0.03  # m, 4 cm duct through the entire depth (z), centered mid-length
WAVESPEED = 4000.0  # m/s,
DENSITY = 2400.0  # kg/m^3, normal-weight concrete
FREQUENCY = 150e3  # 60e3  # Hz
POINTS_PER_WAVELENGTH = 12
TRAVERSALS = 1  # full back-and-forth trips along LENGTH_X to simulate
AMPLITUDE = 1e8
CYCLES = 5
MIN_INDICATOR = 1e-3
MARGIN = 0.01  # m, edge strip excluded from the color-scale max (boundary reflections)
K_FACTOR = (
    1e14  # perturbation weight of u_dagger in the superposed field u^s = u + k u_dagger
)

# --------------------------------------- setup ---------------------------------------
dx_target = WAVESPEED / FREQUENCY / POINTS_PER_WAVELENGTH
LENGTHS = (LENGTH_X, LENGTH_YZ, LENGTH_YZ)
Nx = tuple(round(L / dx_target) + 3 for L in LENGTHS)
dx = tuple(L / (n - 3) for L, n in zip(LENGTHS, Nx))
dt = 0.95 * min(dx) / WAVESPEED / math.sqrt(3)
T = TRAVERSALS * 2 * LENGTH_X / WAVESPEED
N = math.ceil(T / dt)
dofs = math.prod(n - 2 for n in Nx)
print(f"mesh size {Nx}, {dofs:.2e} dofs")

# float64: the superposition trick recovers the cross term by subtracting two close
# quadratic-form evaluations, so it needs more headroom than the direct PML adjoint
sim = scalar_simulation(
    Nx, dx, N, dt, THREADS, precision="float32", wavespeed=WAVESPEED, density=DENSITY
)

x = np.linspace(-dx[0], LENGTH_X + dx[0], Nx[0])
y = np.linspace(-dx[1], LENGTH_YZ + dx[1], Nx[1])
x, y = np.meshgrid(x, y, indexing="ij")
hole = (x - LENGTH_X / 2) ** 2 + (y - LENGTH_YZ / 2) ** 2 < (HOLE_DIAMETER / 2) ** 2

indicator = np.ones(Nx, dtype=np.float64)
indicator[hole] = MIN_INDICATOR

indicator_padded = cp.ones(sim.Nx_padded, dtype=sim.dtype)
indicator_padded[: Nx[0], : Nx[1], : Nx[2]] = cp.asarray(indicator, dtype=sim.dtype)

source_pos = cp.array([[1], [Nx[1] // 2], [Nx[2] // 2]], dtype=cp.int32)
sensor_pos = cp.array([[Nx[0] - 2], [Nx[1] // 2], [Nx[2] // 2]], dtype=cp.int32)


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

# -------------------------------------- gradient --------------------------------------
um = cp.zeros((N, 1), dtype=sim.dtype)  # silent target: adjoint of the recorded energy
subtracted_kernels = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

cp.cuda.Stream.null.synchronize()
tic = time.time()
cost, subtracted_kernels = compute_subtracted_kernels(
    subtracted_kernels, sim, source, indicator_padded, um, sensor_pos, K_FACTOR
)
gradient = compute_sensitivity(sim, subtracted_kernels, K_FACTOR, num_sources=1)
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s, received energy {cost:.2e}")

# ----------------------------------- postprocessing ----------------------------------
interior = tuple(slice(0, n) for n in Nx)
gradient_np = gradient.get()[interior]
z_mid = Nx[2] // 2
field = gradient_np[:, :, z_mid]

margin = (round(MARGIN / dx[0]), round(MARGIN / dx[1]))
scale = (
    float(np.max(np.abs(field[margin[0] : -margin[0], margin[1] : -margin[1]]))) * 0.3
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.pcolormesh(x, y, field, cmap="Spectral_r", vmin=-scale, vmax=scale)
ax.add_patch(
    Circle((LENGTH_X / 2, LENGTH_YZ / 2), HOLE_DIAMETER / 2, fill=False, edgecolor="k")
)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
