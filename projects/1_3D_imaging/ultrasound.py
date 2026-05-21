import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------- settings --------------------------
# physics
C = 1.0
DT = 0.002
T_START = -0.2
T_MAX = 2.0
PULSE_WIDTH = 0.01

R_MAX = 0.5 * (T_MAX - T_START) / C


# sensors / emitter (physical coordinates)
SENSOR_XS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
SENSOR_Y = R_MAX
EMITTER_IDX = 0

# defect (physical coordinates)
DEFECT_X = 0.4
DEFECT_Y = 0.4
DEFECT_SIZE = 0.1

# material grid (resolution only)
NX = 40
NY = 20

print(R_MAX, SENSOR_XS[EMITTER_IDX])

# -------------------- domain + reflectivity -------------------
x = np.linspace(SENSOR_XS[EMITTER_IDX] - R_MAX, SENSOR_XS[EMITTER_IDX] + R_MAX, NX)
y = np.linspace(0, R_MAX, NY)

z = np.zeros((NX, NY))
z[
    np.ix_(
        np.abs(x - DEFECT_X) <= DEFECT_SIZE / 2, np.abs(y - DEFECT_Y) <= DEFECT_SIZE / 2
    )
] = 1.0

# --------------------- sensor geometry ------------------------
emitter_pos = np.array([SENSOR_XS[EMITTER_IDX], SENSOR_Y])

# ---------------------- forward model -------------------------
t = np.arange(T_START, T_MAX, DT)
signals = np.zeros((len(SENSOR_XS), len(t)))


def pulse(tau):
    return np.exp(-(tau**2) / (2 * PULSE_WIDTH**2))


# source
for j, sx in enumerate(SENSOR_XS):
    sensor_pos = np.array([sx, SENSOR_Y])
    tof_direct = np.linalg.norm(sensor_pos - emitter_pos) / C
    signals[j] += pulse(t - tof_direct)

# reflection (Born approximation)
for ix in range(NX):
    for iy in range(NY):
        if z[ix, iy] == 0.0:
            continue
        cell = np.array([x[ix], y[iy]])
        d_emit = np.linalg.norm(cell - emitter_pos)
        for j, sx in enumerate(SENSOR_XS):
            sensor_pos = np.array([sx, SENSOR_Y])
            tof = (d_emit + np.linalg.norm(cell - sensor_pos)) / C
            signals[j] += z[ix, iy] * pulse(t - tof)

# ---------------------------- delay-and-sum -----------------------------
signals_das = signals.copy()
for j, sx in enumerate(SENSOR_XS):
    tof_direct = np.linalg.norm(np.array([sx, SENSOR_Y]) - emitter_pos) / C
    signals_das[j] -= pulse(t - tof_direct)

das = np.zeros((NX, NY))

for ix in range(NX):
    for iy in range(NY):
        cell = np.array([x[ix], y[iy]])
        d_emit = np.linalg.norm(cell - emitter_pos)
        for j, sx in enumerate(SENSOR_XS):
            sensor_pos = np.array([sx, SENSOR_Y])
            tof = (d_emit + np.linalg.norm(cell - sensor_pos)) / C
            das[ix, iy] += np.interp(tof, t, signals_das[j])


# --------------------------- post-processing ----------------------------
xx, yy = np.meshgrid(x, y, indexing="ij")

fig, ax = plt.subplots(figsize=(NX / 10, NY / 10), dpi=100)
ax.imshow(z.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)

fig2, ax2 = plt.subplots(figsize=(NX / 10, NY / 10), dpi=100)
ax2.imshow(das.T, origin="lower", cmap="binary")
ax2.axis("off")
ax2.set_rasterized(True)
fig2.tight_layout(pad=0)

if args.book:
    save_csv(
        RESULTS_DIR / "ultrasound_signals.csv",
        t=t,
        **{f"y{j}": signals[j] for j in range(len(SENSOR_XS))},
    )
    save_csv(
        RESULTS_DIR / "ultrasound_signals_das.csv",
        t=t,
        **{f"y{j}": signals_das[j] for j in range(len(SENSOR_XS))},
    )
    fig.savefig(RESULTS_DIR / "ultrasound_groundtruth.png")
    fig2.savefig(RESULTS_DIR / "ultrasound_prediction.png")
    plt.close("all")
else:
    plt.show()
