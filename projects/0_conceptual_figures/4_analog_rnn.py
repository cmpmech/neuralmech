import argparse
from dataclasses import replace
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuwave.boundary import pad_for_sponge, sponge
from cuwave.utils import Sensors, point_source
from cuwave.wave import AcousticWave, simulate, stable_dt

from postprocessing import cmyk_to_rgb

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# geometry
LENGTHS = (200.0, 100.0)
SOURCE = [(10.0, 50.0)]
SENSOR = [(190.0, 25.0), (190.0, 50.0), (190.0, 75.0)]

# discretization
RESOLUTION = (3000, 1500)
SPACE_ORDER = 2  # finite difference order, any even number
SAFETY = 0.9  # fraction of the stable time step
THREADS = (4, 64)
T = 1.5

# physics: air only, so the wave spreads without a scatterer
RHO1 = 1.204
KAPPA1 = 1.419e5
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10

# absorbing sponge on every edge
SPONGE_THICKNESS = 3.35  # metres, so it does not need rescaling with RESOLUTION
SPONGE_BETA = 0.1  # peak damping d * dt / 2m at the wall

# postprocessing
RECORD_EVERY = 10  # animation frame rate
SNAPSHOT = 0.55  # seconds into the run the still snapshot is drawn at

# --------------------------------------- setup ---------------------------------------
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(2))
Nx, width, origin, region = pad_for_sponge(RESOLUTION, dx, SPONGE_THICKNESS)
x0, y0 = origin  # where the region of interest starts, the sponge sitting before it

wavespeed = np.sqrt(KAPPA1 / RHO1)
dt = SAFETY * stable_dt(dx, wavespeed, SPACE_ORDER)
N = int(T / dt)
f_max = wavespeed / (POINTS_PER_WAVELENGTH * max(dx))

sim = AcousticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision="float32",
    space_order=SPACE_ORDER,
    rho1=RHO1,
    rho2=RHO1,  # only air
    kappa1=KAPPA1,
    kappa2=KAPPA1,  # only air
)
# damping sets a compile flag, so it has to be in place before any kernel compiles
sim = replace(
    sim, damping=sponge(sim, cp.ones(sim.Nx_padded, dtype=sim.dtype), width, SPONGE_BETA)
)

# material distribution: uniform air, the design of the trained medium left out
gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

sensors = Sensors(sim, [(x0 + x, y0 + y) for x, y in SENSOR])
source_coords = [(x0 + x, y0 + y) for x, y in SOURCE]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "minecraft_mobs.npz")
creeper_ids = np.where(data["mob"] == "creeper")[0]
if len(creeper_ids) == 0:
    raise SystemExit(
        "no creeper clip in minecraft_mobs.npz; run minecraft_mobs_download.py first"
    )
clip = data["X"][creeper_ids[0]]

# compress the clip's full spectrum into the resolvable band [0, f_max], then resample
# onto the simulation time base
bins = int(2.0 * f_max * T) // 2 + 1
source_wave = np.fft.irfft(np.fft.rfft(clip)[:bins], N)
source_wave = source_wave / np.max(np.abs(source_wave))
source = point_source(sim, source_coords, AMPLITUDE * source_wave)

# -------------------------------------- simulate -------------------------------------
# animate: store every RECORD_EVERY steps; otherwise store only the snapshot step
snapshot_step = min(max(int(SNAPSHOT / dt), 1), N - 1)
record_every = RECORD_EVERY if args.animate else snapshot_step
_, um, frames = simulate(
    sim, source, gamma, sensors=sensors.nodes, record_every=record_every
)
traces = sensors.traces(um).get()
t = np.linspace(0, (N - 1) * dt, N)

# ----------------------------------- postprocessing ----------------------------------
if not args.animate:
    # wavefield snapshot
    snap = frames[1][region]
    scale = float(np.max(np.abs(snap))) * 0.5
    fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=100)
    ax.imshow(snap.T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale)
    ax.set_rasterized(True)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if not args.book:
        plt.show()
    else:
        fig.savefig(RGB_PDF_DIR / "analog_rnn_field.pdf")
        plt.close()

    # source
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    scale = np.max(np.abs(source_wave))
    ax.set_ylim(-scale, scale)
    ax.set_xlim(0, T)
    ax.plot(t, source_wave, color=cmyk_to_rgb(0, 0.76, 0.8, 0.2), linewidth=1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if not args.book:
        plt.show()
    else:
        fig.savefig(RGB_PDF_DIR / "analog_rnn_source.pdf", transparent=True)
        plt.close()

    # sensor signals: sensor k is the class-k readout
    scale = np.max(np.abs(traces))
    for k in range(len(SENSOR)):
        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        ax.set_ylim(-scale, scale)
        ax.set_xlim(0, T)
        ax.plot(t, traces[:, k], color=cmyk_to_rgb(0.8, 0.44, 0, 0.2), linewidth=1)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if not args.book:
            plt.show()
        else:
            fig.savefig(RGB_PDF_DIR / f"analog_rnn_sensors_{k}.pdf", transparent=True)
            plt.close()
# ----------------------------------- animate export ----------------------------------
else:
    ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/analog_rnn"
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    scale = float(np.max(np.abs(frames[:, region[0], region[1]]))) * 0.2
    for f, frame in enumerate(frames):
        fig, ax = plt.subplots(
            figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150
        )
        ax.imshow(
            frame[region].T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(ANIMATION_DIR / f"frame_{f:04d}.jpg")
        plt.close()
