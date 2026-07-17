import math
import time

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from solvers.wave import elastic_simulation, setup_source, simulate

# -------------------------------------- settings -------------------------------------
THREADS = (4, 4, 64)

# physics
LENGTH_X = 0.3  # m, 30 cm block length
LENGTH_YZ = 0.15  # m, 15x15 cm cross section
HOLE_DIAMETER = 0.03  # m, 3 cm duct through the entire depth (z), centered mid-length
YOUNGS_MODULUS = 30e9  # Pa, normal-weight concrete
POISSON = 0.2
DENSITY = 2400.0  # kg/m^3
FREQUENCY = 150e3  # Hz
POINTS_PER_WAVELENGTH = 12  # resolved on the shorter shear wavelength
TRAVERSALS = 1  # full back-and-forth trips along LENGTH_X to simulate
AMPLITUDE = 1e8
CYCLES = 5
MIN_INDICATOR = 1e-3
CFL = 0.9

# --------------------------------------- setup ---------------------------------------
LAME_MU = YOUNGS_MODULUS / (2 * (1 + POISSON))
LAME_LAMBDA = YOUNGS_MODULUS * POISSON / ((1 + POISSON) * (1 - 2 * POISSON))
WAVESPEED_P = math.sqrt((LAME_LAMBDA + 2 * LAME_MU) / DENSITY)  # pressure wave, fastest
WAVESPEED_S = math.sqrt(LAME_MU / DENSITY)  # shear wave, shortest wavelength

dx_target = WAVESPEED_S / FREQUENCY / POINTS_PER_WAVELENGTH
LENGTHS = (LENGTH_X, LENGTH_YZ, LENGTH_YZ)
Nx = tuple(round(L / dx_target) + 3 for L in LENGTHS)
dx = tuple(L / (n - 3) for L, n in zip(LENGTHS, Nx))
dt = CFL * min(dx) / WAVESPEED_P / math.sqrt(3)
T = TRAVERSALS * 2 * LENGTH_X / WAVESPEED_P
N = math.ceil(T / dt)
dofs = 3 * math.prod(n - 2 for n in Nx)
print(
    f"mesh size {Nx}, c_p {WAVESPEED_P:.0f} m/s, c_s {WAVESPEED_S:.0f} m/s, {dofs:.2e} dofs"
)

sim = elastic_simulation(
    Nx, dx, N, dt, THREADS, density=DENSITY, lame_lambda=LAME_LAMBDA, lame_mu=LAME_MU
)

x = np.linspace(-dx[0], LENGTH_X + dx[0], Nx[0])
y = np.linspace(-dx[1], LENGTH_YZ + dx[1], Nx[1])
x, y = np.meshgrid(x, y, indexing="ij")
hole = (x - LENGTH_X / 2) ** 2 + (y - LENGTH_YZ / 2) ** 2 < (HOLE_DIAMETER / 2) ** 2

indicator = np.ones(Nx, dtype=np.float32)
indicator[hole] = MIN_INDICATOR

indicator_padded = cp.ones(sim.Nx_padded, dtype=sim.dtype)
indicator_padded[: Nx[0], : Nx[1], : Nx[2]] = cp.asarray(indicator, dtype=sim.dtype)

# longitudinal transducer on the near face: a point body force along x (component 0)
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
source = setup_source(source_pos, signal, component=0)

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
field = u_np[0, :, :, z_mid]  # longitudinal displacement u_x on the mid slice
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
