from pathlib import Path

import cmasher as cmr
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from analog_rnn_fixture import (
    DATASET,
    MATERIAL,
    RESOLUTION,
    SENSOR,
    N,
    dt,
    load_source,
    probabilities,
    region,
    sensors,
    sim,
)
from cuwave.wave import simulate

from postprocessing import cmyk_to_rgb

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

# -------------------------------------- settings -------------------------------------
CLASS = 0  # the first clip of this class is the one played through the medium
SNAPSHOT = 1.35  # seconds into the run the field is drawn at
DESIGN = True  # False plays the same clip through free field, as the reference
SATURATION = 0.2  # fraction of the peak the field colormap runs to

# ------------------------------------- load model ------------------------------------
if not MATERIAL.exists():
    raise SystemExit(f"no trained material at {MATERIAL}; run analog_rnn_train.py first")
material = np.load(MATERIAL) & DESIGN
gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
gamma[region] = cp.asarray(material, dtype=sim.dtype)

# ------------------------------------- load data -------------------------------------
data = np.load(DATASET)
clips = np.where(data["y"] == CLASS)[0]
if len(clips) == 0:
    raise SystemExit(f"no clip of class {CLASS} in {DATASET.name}")
source = load_source(data["X"][clips[0]])

# -------------------------------------- simulate -------------------------------------
# every record_every-th step is snapshotted, so frame 1 is the one at SNAPSHOT
step = min(max(int(SNAPSHOT / dt), 1), N - 1)
_, um, frames = simulate(sim, source, gamma, sensors=sensors.nodes, record_every=step)
traces = sensors.traces(um).get()
probs = probabilities(sensors.traces(um)).get()

print(f"clip {data['files'][clips[0]]} ({data['mob'][clips[0]]})")
print({str(name): round(float(p), 3) for name, p in zip(data["classes"], probs)})
print(f"predicted {data['classes'][int(probs.argmax())]}")

# ----------------------------------- postprocessing ----------------------------------
# wavefield snapshot with the trained scatterer overlaid
snap = frames[1][region]
if DESIGN:
    scale = float(np.max(np.abs(snap[~material]))) * SATURATION
else:
    scale = float(np.max(np.abs(snap))) * 0.5
overlay = np.ma.masked_where(~material, material.astype(float))
fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150)
ax.imshow(snap.T, origin="lower", cmap=cmr.fusion, vmin=-scale, vmax=scale)
ax.imshow(overlay.T, origin="lower", cmap="binary", vmin=0, vmax=1, alpha=1)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if DESIGN:
    fig.savefig(RGB_PDF_DIR / f"analog_rnn_eval_field_{CLASS}.pdf")
else:
    fig.savefig(RGB_PDF_DIR / "analog_rnn_field.pdf")
plt.close()

# source signal
t = np.linspace(0, (N - 1) * dt, N)
source_wave = source.signal.sum(axis=1).get()
fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
scale = np.max(np.abs(source_wave))
ax.set_ylim(-scale, scale)
ax.set_xlim(0, t[-1])
ax.plot(t, source_wave, color=cmyk_to_rgb(0, 0.76, 0.8, 0.2), linewidth=1)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if DESIGN:
    fig.savefig(RGB_PDF_DIR / f"analog_rnn_eval_source_{CLASS}.pdf", transparent=True)
else:
    fig.savefig(RGB_PDF_DIR / "analog_rnn_source.pdf", transparent=True)
plt.close()

# sensor signals: sensor k is the class-k readout
scale = np.max(np.abs(traces))
for k in range(len(SENSOR)):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.set_ylim(-scale, scale)
    ax.set_xlim(0, t[-1])
    ax.plot(t, traces[:, k], color=cmyk_to_rgb(0.8, 0.44, 0, 0.2), linewidth=1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if DESIGN:
        fig.savefig(
            RGB_PDF_DIR / f"analog_rnn_eval_sensor_{CLASS}_{k}.pdf", transparent=True
        )
    else:
        fig.savefig(RGB_PDF_DIR / f"analog_rnn_sensors_{k}.pdf", transparent=True)
    plt.close()
