import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# sensors (physical coordinates)
SENSOR_XS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# SENSOR_XS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
SENSOR_Y = R_MAX

# defect (physical coordinates)
DEFECT_X = 0.4
DEFECT_Y = 0.4
DEFECT_SIZE = 0.1

# material grid (resolution only)
NX = 100
NY = 50

# -------------------- domain + reflectivity -------------------
x = np.linspace(SENSOR_XS.mean() - R_MAX, SENSOR_XS.mean() + R_MAX, NX)
y = np.linspace(0, R_MAX, NY)

z = np.zeros((NX, NY))
z[
    np.ix_(
        np.abs(x - DEFECT_X) <= DEFECT_SIZE / 2, np.abs(y - DEFECT_Y) <= DEFECT_SIZE / 2
    )
] = 1.0

# ---------------------- forward model -------------------------
t = np.arange(T_START, T_MAX, DT)
n = len(SENSOR_XS)
signals = np.zeros((n, n, len(t)))


def pulse(tau):
    return np.exp(-(tau**2) / (2 * PULSE_WIDTH**2))


for e, ex in enumerate(SENSOR_XS):
    emitter_pos = np.array([ex, SENSOR_Y])

    # source
    for j, sx in enumerate(SENSOR_XS):
        tof_direct = np.linalg.norm(np.array([sx, SENSOR_Y]) - emitter_pos) / C
        signals[e, j] += pulse(t - tof_direct)

    # reflection (Born approximation)
    for ix in range(NX):
        for iy in range(NY):
            if z[ix, iy] == 0.0:
                continue
            cell = np.array([x[ix], y[iy]])
            d_emit = np.linalg.norm(cell - emitter_pos)
            for j, sx in enumerate(SENSOR_XS):
                tof = (d_emit + np.linalg.norm(cell - np.array([sx, SENSOR_Y]))) / C
                signals[e, j] += z[ix, iy] * pulse(t - tof)

# ---------------------------- delay-and-sum -----------------------------
signals_das = signals.copy()
for e, ex in enumerate(SENSOR_XS):
    emitter_pos = np.array([ex, SENSOR_Y])
    for j, sx in enumerate(SENSOR_XS):
        tof_direct = np.linalg.norm(np.array([sx, SENSOR_Y]) - emitter_pos) / C
        signals_das[e, j] -= pulse(t - tof_direct)

das = np.zeros((NX, NY))
for ix in range(NX):
    for iy in range(NY):
        cell = np.array([x[ix], y[iy]])
        for e, ex in enumerate(SENSOR_XS):
            d_emit = np.linalg.norm(cell - np.array([ex, SENSOR_Y]))
            for j, sx in enumerate(SENSOR_XS):
                tof = (d_emit + np.linalg.norm(cell - np.array([sx, SENSOR_Y]))) / C
                das[ix, iy] += np.interp(tof, t, signals_das[e, j])

# --------------------------- post-processing ----------------------------
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "ultrasoundV2_groundtruth.png")
    fig2.savefig(RESULTS_DIR / "ultrasoundV2_prediction.png")
    plt.close("all")
else:
    plt.show()
