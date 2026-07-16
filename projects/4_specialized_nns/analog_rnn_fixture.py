import math

import cupy as cp
import numpy as np
from scipy.signal import resample

from solvers.wave import acoustic_simulation, build_sponge

# -------------------------------------- settings -------------------------------------
# geometry: source on the left wall, three probes on the right wall (one per class)
LENGTHS = (200.0, 100.0)
SOURCE = [(10.0, 50.0)]
SENSOR = [(190.0, 25.0), (190.0, 50.0), (190.0, 75.0)]

# discretization
# below ~(2800, 1400) the sensor energy keeps shrinking under grid refinement instead
# of converging, since RMIN-sized features are only ~4 cells wide
RESOLUTION = (3000, 1500)
CFL = 0.9
T = 1.5

# physics: air (material 1) and a light polymer foam scatterer (material 2). Foam's
# low acoustic impedance (~13x air, vs ~10000x for aluminium) lets sound transmit into
# the design rather than mirror off it, so the probe signals keep their magnitude and
# bandwidth (slab diagnostic: ~52% of free-field RMS transmitted, vs 0.7% for aluminium)
RHO1, RHO2 = 1.204, 30.0
KAPPA1, KAPPA2 = 1.419e5, 9.72e5  # foam bulk modulus, c2 ~ 180 m/s
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10

# absorbing sponge on every edge [x-, x+, y-, y+] so probe energies are not degenerate
BOUNDARIES = ["pml", "pml", "pml", "pml"]
SPONGE_WIDTH = 40  # scale alongside RESOLUTION
SPONGE_BETA = 0.1

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


# --------------------------------------- helper --------------------------------------
def load_source(clip):
    # compress the clip's full spectrum into the resolvable band [0, f_max] (the raw
    # clips carry almost no energy below f_max, so low-passing would keep only
    # amplified filter residue), then sinc-interpolate onto the simulation time base
    n_band = int(2.0 * f_max * T)
    wave = resample(resample(clip, n_band), N)
    wave = wave / np.max(np.abs(wave))
    return cp.asarray(AMPLITUDE * wave[:, None] / np.prod(dx), dtype=sim.dtype)
