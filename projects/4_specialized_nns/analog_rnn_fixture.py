import math

import cupy as cp
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from solvers.wave import acoustic_simulation, build_sponge

# -------------------------------------- settings -------------------------------------
# geometry: source on the left wall, three probes on the right wall (one per class)
LENGTHS = (200.0, 100.0)
SOURCE = [(10.0, 50.0)]
SENSOR = [(190.0, 25.0), (190.0, 50.0), (190.0, 75.0)]

# discretization
# RESOLUTION = (96, 48)
# RESOLUTION = (200, 100)
RESOLUTION = (3000, 1500)
CFL = 0.9  # 0.5
T = 1.5

# physics: air (material 1) and a moderate-contrast dense scatterer (material 2)
RHO1, RHO2 = 1.204, 12.04
KAPPA1, KAPPA2 = 1.419e5, 1.419e5
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10

# absorbing sponge on every edge [x-, x+, y-, y+] so probe energies are not degenerate
BOUNDARIES = ["pml", "pml", "pml", "pml"]
SPONGE_WIDTH = 50  # matches 4_analog_rnn.py; 8 also holds at RESOLUTION <= (2400, 1200)
SPONGE_BETA = 0.1  # matches 4_analog_rnn.py; 1.5 also holds at RESOLUTION <= (2400, 1200)

# --------------------------------------- setup ---------------------------------------
# increase grid by sponge layer on each absorbing edge
pad_lo = tuple(SPONGE_WIDTH if BOUNDARIES[2 * d] == "pml" else 0 for d in range(2))
pad_hi = tuple(SPONGE_WIDTH if BOUNDARIES[2 * d + 1] == "pml" else 0 for d in range(2))

# spatial grid
Nx = tuple(RESOLUTION[d] + pad_lo[d] + pad_hi[d] for d in range(2))
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(2))
# helpers
to_index = lambda coord: tuple(
    int(round(coord[d] / dx[d])) + pad_lo[d] for d in range(2)
)
crop = tuple(slice(1 + pad_lo[d], Nx[d] - 1 - pad_hi[d]) for d in range(2))

# temporal grid
wavespeeds = np.sqrt(np.array([KAPPA2 / RHO2, KAPPA1 / RHO1]))
dt = CFL * min(dx) / np.max(wavespeeds) / math.sqrt(2)
N = math.ceil(T / dt)

f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))
print(f"steps {N}, max resolvable frequency {f_max:.1f} Hz")

sim = acoustic_simulation(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision="float32",
    rho1=RHO1,
    rho2=RHO2,
    kappa1=KAPPA1,
    kappa2=KAPPA2,
)

# boundary conditions
EDGES = [(0, "lo"), (0, "hi"), (1, "lo"), (1, "hi")]
sides = [e for e, b in zip(EDGES, BOUNDARIES) if b == "pml"]
sponge = build_sponge(
    sim, SPONGE_WIDTH, 2.0 * SPONGE_BETA / (KAPPA1 * dt), power=3, sides=sides
)

# source and probes: probe m is the class-m readout (hostile 0, neutral 1, passive 2)
srcs = [to_index((x, y)) for x, y in SOURCE]
source_position = cp.array([[s[0] for s in srcs], [s[1] for s in srcs]], dtype=cp.int32)
cols = [to_index((x, y)) for x, y in SENSOR]
sensors = cp.array([[c[0] for c in cols], [c[1] for c in cols]], dtype=cp.int32)


# -------------------------------------- helper ---------------------------------------
def load_source(clip):
    # resample the clip onto the simulation time base, low-pass to the grid's max
    # resolvable frequency (zero-phase), normalize, and scale to the source amplitude
    sos = butter(8, f_max, btype="low", fs=1.0 / dt, output="sos")
    wave = sosfiltfilt(sos, resample(clip, N))
    wave = wave / np.max(np.abs(wave))
    return cp.asarray(AMPLITUDE * wave[:, None] / np.prod(dx), dtype=sim.dtype)
