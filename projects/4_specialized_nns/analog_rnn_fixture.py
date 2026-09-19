import math
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np
from cuwave.boundary import pad_for_sponge, sponge
from cuwave.scalar import AcousticWave
from cuwave.sensitivity import reconstruction_nodes
from cuwave.utils import Sensors, point_source
from cuwave.wave import stable_dt

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODELS_DIR = (BASE_DIR / "../../models").resolve()

DATASET = DATA_DIR / "minecraft_mobs.npz"
MATERIAL = MODELS_DIR / "analog_rnn_material.npy"

# -------------------------------------- settings -------------------------------------
# geometry: source on the left wall, three probes on the right wall (one per class)
LENGTHS = (200.0, 100.0)
SOURCE = [(10.0, 50.0)]
SENSOR = [(190.0, 25.0), (190.0, 50.0), (190.0, 75.0)]

# discretization
RESOLUTION = (3000, 1500)  # 600 x 300 and 1200 x 600 fit too, at 1/125 and 1/16 the time
SPACE_ORDER = 2  # above 2 the adjoint is only consistent, and a binary design is rough
PRECISION = "float32"
THREADS = (4, 64)
SAFETY = 0.99  # fraction of the stable time step
T = 1.5  # matches the clip length the dataset is padded to

# physics: air (material 1) and a light polymer foam scatterer (material 2). Foam's
# low acoustic impedance (~13x air, vs ~10000x for aluminium) lets sound transmit into
# the design rather than mirror off it, so the probe signals keep their magnitude and
# bandwidth (slab diagnostic: ~52% of free-field RMS transmitted, vs 0.7% for aluminium)
RHO1, RHO2 = 1.204, 30.0
KAPPA1, KAPPA2 = 1.419e5, 9.72e5  # foam bulk modulus, c2 ~ 180 m/s
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10

# absorbing sponge on every edge so the probe energies are not degenerate
SPONGE_THICKNESS = 10.0  # metres, so it does not need rescaling with RESOLUTION
SPONGE_BETA = 0.05  # peak damping d * dt / 2m at the wall

# --------------------------------------- setup ---------------------------------------
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(2))
Nx, width, origin, region = pad_for_sponge(RESOLUTION, dx, SPONGE_THICKNESS)
x0, y0 = origin  # where the region of interest starts, the sponge sitting before it

wavespeeds = np.sqrt(np.array([KAPPA1 / RHO1, KAPPA2 / RHO2]))
dt = SAFETY * stable_dt(dx, np.max(wavespeeds), SPACE_ORDER)
N = math.ceil(T / dt)
f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))

sim = AcousticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    rho1=RHO1,
    rho2=RHO2,
    kappa1=KAPPA1,
    kappa2=KAPPA2,
)
# damping sets a compile flag, so it has to be in place before any kernel compiles
sim = replace(
    sim, damping=sponge(sim, cp.ones(sim.Nx_padded, dtype=sim.dtype), width, SPONGE_BETA)
)

# node index of a coordinate of the region of interest, the sponge offset added
to_node = lambda coord: tuple(
    int(round((origin[d] + coord[d]) / dx[d])) + 1 for d in range(2)
)

# probe m is the class-m readout (hostile 0, neutral 1, passive 2)
source_coords = [(x0 + x, y0 + y) for x, y in SOURCE]
sensors = Sensors(sim, [(x0 + x, y0 + y) for x, y in SENSOR])

strip, _ = reconstruction_nodes(sim)
footprint = (N + 2) * strip.shape[1] * np.dtype(sim.dtype).itemsize
print(f"{Nx[0]} x {Nx[1]} nodes, {N} steps, {f_max:.0f} Hz max resolvable")
print(f"adjoint strip {strip.shape[1]} nodes, {footprint / 1e6:.0f} MB")


# -------------------------------------- helper ---------------------------------------
def load_source(clip):
    # compress the clip's full spectrum into the resolvable band [0, f_max] (the raw
    # clips carry almost no energy below f_max, so low-passing would keep only
    # amplified filter residue), then resample onto the simulation time base
    bins = int(2.0 * f_max * T) // 2 + 1
    wave = np.fft.irfft(np.fft.rfft(clip)[:bins], N)
    return point_source(sim, source_coords, AMPLITUDE * wave / np.max(np.abs(wave)))


def probabilities(traces):
    # normalized probe energy y_m = sum_t u_m^2, equivalently a softmax over log-energies
    y = cp.sum(traces**2, axis=0)
    return y / cp.sum(y)


def cross_entropy(label, penalty=0.0):
    # loss -log p[label] - penalty * log(sum y), returned with its derivative dJ/dtraces.
    # the penalty rewards energy reaching the probes at all rather than only how it
    # splits across them; the log keeps it dimensionless and on the same footing as the
    # softmax normalization regardless of the raw energy scale (which swings over many
    # orders of magnitude as the design binarizes)
    def objective(traces):
        y = cp.sum(traces**2, axis=0)
        total = float(cp.sum(y))
        cost = -math.log(float(y[label]) / total) - penalty * math.log(total)
        # dJ/dy_m, carried onto the traces by dy_m/du_m = 2 u_m
        dy = cp.full(y.shape, (1.0 - penalty) / total, dtype=traces.dtype)
        dy[label] -= 1.0 / float(y[label])
        return cost, 2.0 * dy * traces

    return objective
