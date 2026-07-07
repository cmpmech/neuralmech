import argparse
import math
from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from postprocessing import cmyk_to_rgb
from solvers.wave import acoustic_simulation, build_sponge, setup_source, simulate

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
SOURCE = [(10, 50)]
SENSOR = [(190, 25), (190, 50), (190, 75)]

# discretization
RESOLUTION = (3000, 1500)
# RESOLUTION = (1500, 750)
CFL = 0.9  # 0.5
T = 1.5

# physics
RHO1 = 1.204
KAPPA1 = 1.419e5
AMPLITUDE = 1e3

# absorbing sponge on every edge [x-, x+, y-, y+]
BOUNDARIES = ["pml", "pml", "pml", "pml"]
SPONGE_WIDTH = 50
SPONGE_BETA = 0.1

# postprocessing
RECORD_EVERY = 10  # animation frame rate
SNAPSHOT_STEP = 4000  # absolute time step of the still snapshot

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
wavespeed = math.sqrt(KAPPA1 / RHO1)
dt = CFL * min(dx) / wavespeed / math.sqrt(2)
N = math.ceil(T / dt)

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

# material distribution
gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)

# source (creeper)
data = np.load(DATA_DIR / "minecraft_mobs.npz")
creeper_ids = np.where(data["mob"] == "creeper")[0]
if len(creeper_ids) == 0:
    raise SystemExit(
        "no creeper clip in minecraft_mobs.npz; run minecraft_mobs_download.py first"
    )
clip = data["X"][creeper_ids[0]]  # audio
sample_rate = int(data["sr"])

f_max = 500  # computed with wavespeed / (10 * max(dx))
# low pass filtering
sos = butter(8, f_max, btype="low", fs=1 / dt, output="sos")  # filter
raw_wave = resample(clip, N)
source_wave = sosfiltfilt(sos, raw_wave)  # apply filter
raw_wave = raw_wave / np.max(np.abs(raw_wave))
source_wave = source_wave / np.max(np.abs(source_wave))

srcs = [to_index((x, y)) for x, y in SOURCE]
position = cp.array([[s[0] for s in srcs], [s[1] for s in srcs]], dtype=cp.int32)
signal = cp.asarray(
    (AMPLITUDE * source_wave / np.prod(dx))[:, None] * np.ones(len(SOURCE)),
    dtype=sim.dtype,
)
source = setup_source(position, signal)

# sensors
cols = [to_index((x, y)) for x, y in SENSOR]
sensors = cp.array([[c[0] for c in cols], [c[1] for c in cols]], dtype=cp.int32)

# -------------------------------------- simulate -------------------------------------
# animate: store every RECORD_EVERY steps; otherwise store only the snapshot step
record_every = RECORD_EVERY if args.animate else SNAPSHOT_STEP
_, um, frames = simulate(
    sim, source, gamma, damping=sponge, sensors=sensors, record_every=record_every
)
um = um.get()
t = np.linspace(0, (N - 1) * dt, N)

# ----------------------------------- postprocessing ----------------------------------
if not args.animate:
    # wavefield snapshot
    snap = frames[SNAPSHOT_STEP // record_every][crop]
    scale = float(np.max(np.abs(snap))) * 0.5
    fig_field, ax_field = plt.subplots(
        figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=100
    )
    ax_field.imshow(snap.T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale)
    ax_field.set_rasterized(True)
    ax_field.axis("off")
    fig_field.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if not args.book:
        plt.show()
    else:
        fig_field.savefig(RGB_PDF_DIR / "analog_rnn_field.pdf")
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

    for k in range(len(SENSOR)):
        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        scale = np.max(np.abs(um[:, k]))
        ax.set_ylim(-scale, scale)
        ax.set_xlim(0, T)
        ax.plot(t, um[:, k], color=cmyk_to_rgb(0.8, 0.44, 0, 0.2), linewidth=1)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if not args.book:
            plt.show()
        else:
            fig.savefig(RGB_PDF_DIR / f"analog_rnn_sensors_{k}.pdf", transparent=True)
            plt.close()

ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/analog_rnn"
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    scale = float(np.max(np.abs(frames[:, crop[0], crop[1]]))) * 0.2
    for f, frame in enumerate(frames):
        fig, ax = plt.subplots(
            figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150
        )
        ax.imshow(
            frame[crop].T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(ANIMATION_DIR / f"frame_{f:04d}.jpg")
        plt.close()
