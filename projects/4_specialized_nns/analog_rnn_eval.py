from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from analog_rnn_fixture import (
    RESOLUTION,
    SENSOR,
    N,
    crop,
    dt,
    load_source,
    sensors,
    sim,
    source_position,
    sponge,
)

from postprocessing import cmyk_to_rgb, save_temp_fig
from solvers.wave import setup_source, simulate

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
MODELS_DIR = (BASE_DIR / "../../models").resolve()

# -------------------------------------- settings -------------------------------------
CLASS = 0
SNAPSHOT_STEP = 10000  # 7200 #11000 #1000 #2500 #1500 #15000 #15000  # 4000  # 10000

DESIGN = True
# ------------------------------------- load model ------------------------------------
material_path = MODELS_DIR / "analog_rnn_material.npy"
if not material_path.exists():
    raise SystemExit("no trained material; run analog_rnn_train.py first")
material = np.load(material_path)
if DESIGN == False:
    material *= False

# DEBUGGING START
# material = np.zeros((2998, 1498), dtype=np.bool)
# material[1500:1600, :] = True
# DEBUGGING END

gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
gamma[crop] = cp.asarray(material, dtype=sim.dtype)


# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "minecraft_mobs.npz")
ids = np.where(data["y"] == CLASS)[0]
if len(ids) == 0:
    raise SystemExit(f"no clip for class {CLASS} in minecraft_mobs.npz")

signal = load_source(data["X"][ids[0]])
source = setup_source(source_position, signal)
source_wave = signal[:, 0].get()


# -------------------------------------- simulate -------------------------------------
record_every = min(SNAPSHOT_STEP, N - 1)
_, um, frames = simulate(
    sim, source, gamma, damping=sponge, sensors=sensors, record_every=record_every
)
um = um.get()
t = np.linspace(0, (N - 1) * dt, N)

# predicted class probabilities: normalized integrated probe energy per sensor
probs = np.sum(um**2, axis=0)
probs /= probs.sum()
print("predicted probabilities:", dict(zip(data["classes"], np.round(probs, 3))))

# ----------------------------------- postprocessing ----------------------------------
# wavefield snapshot with the trained scatterer overlaid (frame 1 is t = record_every)
snap = frames[1][crop]
if DESIGN == True:
    scale = float(np.max(np.abs(snap[~material]))) * 0.2
else:
    scale = float(np.max(np.abs(snap))) * 0.5
overlay = np.ma.masked_where(~material, material.astype(float))
fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150)
ax.imshow(snap.T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.imshow(overlay.T, origin="lower", cmap="binary", vmin=0, vmax=1, alpha=1)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if DESIGN == True:
    fig.savefig(RGB_PDF_DIR / f"analog_rnn_eval_field_{CLASS}.pdf")
else:
    fig.savefig(RGB_PDF_DIR / "analog_rnn_field.pdf")
# save_temp_fig(RESULTS_DIR / f"analog_rnn_eval_field_{CLASS}")
plt.close()

# source signal
fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
scale = np.max(np.abs(source_wave))
ax.set_ylim(-scale, scale)
ax.set_xlim(0, t[-1])
ax.plot(t, source_wave, color=cmyk_to_rgb(0, 0.76, 0.8, 0.2), linewidth=1)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if DESIGN == True:
    fig.savefig(RGB_PDF_DIR / f"analog_rnn_eval_source_{CLASS}.pdf", transparent=True)
else:
    fig.savefig(RGB_PDF_DIR / "analog_rnn_source.pdf", transparent=True)
# save_temp_fig(RESULTS_DIR / f"analog_rnn_eval_source_{CLASS}")
plt.close()

# sensor signals: sensor k is the class-k readout
scale = np.max(np.abs(um))
for k in range(len(SENSOR)):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.set_ylim(-scale, scale)
    ax.set_xlim(0, t[-1])
    ax.plot(t, um[:, k], color=cmyk_to_rgb(0.8, 0.44, 0, 0.2), linewidth=1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if DESIGN == True:
        fig.savefig(
            RGB_PDF_DIR / f"analog_rnn_eval_sensor_{CLASS}_{k}.pdf", transparent=True
        )
    else:
        fig.savefig(RGB_PDF_DIR / f"analog_rnn_sensors_{k}.pdf", transparent=True)
    # save_temp_fig(RESULTS_DIR / f"analog_rnn_eval_sensor_{CLASS}_{k}")
    plt.close()
