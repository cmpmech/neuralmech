import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper import born_forward, das_backproject
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
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

# -------------------------------------- geometry -------------------------------------
x = np.linspace(SENSOR_XS[EMITTER_IDX] - R_MAX, SENSOR_XS[EMITTER_IDX] + R_MAX, NX)
y = np.linspace(0, R_MAX, NY)

z = np.zeros((NX, NY))
# reflective defect
z[
    np.ix_(
        np.abs(x - DEFECT_X) <= DEFECT_SIZE / 2, np.abs(y - DEFECT_Y) <= DEFECT_SIZE / 2
    )
] = 1.0

# sensor geometry
sensor_positions = np.stack([SENSOR_XS, np.full_like(SENSOR_XS, SENSOR_Y)], axis=1)
emitter_pos = sensor_positions[EMITTER_IDX]

# ----------------------------------- forward model -----------------------------------
t = np.arange(T_START, T_MAX, DT)


def pulse(tau):
    return np.exp(-(tau**2) / (2 * PULSE_WIDTH**2))


# reflection (Born approximation)
signals = born_forward(emitter_pos, sensor_positions, z, x, y, t, C, pulse)

# ----------------------------------- reconstruction ----------------------------------
# delay-and-sum: remove the direct arrival, then backproject
signals_das = signals.copy()
for j, sensor_pos in enumerate(sensor_positions):
    tof_direct = np.linalg.norm(sensor_pos - emitter_pos) / C
    signals_das[j] -= pulse(t - tof_direct)

das = das_backproject(emitter_pos, sensor_positions, signals_das, x, y, t, C)


# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(NX / 10, NY / 10), dpi=100)
ax.imshow(z.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    fig.savefig(RESULTS_DIR / "ultrasound_groundtruth.png")
else:
    plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(NX / 10, NY / 10), dpi=100)
ax.imshow(das.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    fig.savefig(RESULTS_DIR / "ultrasound_prediction.png")
else:
    plt.show()
plt.close(fig)

# -------------------------------- book postprocessing --------------------------------
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
