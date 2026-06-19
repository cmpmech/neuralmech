import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from helper import born_forward, das_backproject

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

# sensors (physical coordinates)
SENSOR_XS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
SENSOR_Y = R_MAX

# defect (physical coordinates)
DEFECT_X = 0.4
DEFECT_Y = 0.4
DEFECT_SIZE = 0.1

# material grid (resolution only)
NX = 100
NY = 50

# -------------------------------------- geometry -------------------------------------
x = np.linspace(SENSOR_XS.mean() - R_MAX, SENSOR_XS.mean() + R_MAX, NX)
y = np.linspace(0, R_MAX, NY)

z = np.zeros((NX, NY))
# reflective defect
z[
    np.ix_(
        np.abs(x - DEFECT_X) <= DEFECT_SIZE / 2, np.abs(y - DEFECT_Y) <= DEFECT_SIZE / 2
    )
] = 1.0

# all sensors emit now
sensor_positions = np.stack([SENSOR_XS, np.full_like(SENSOR_XS, SENSOR_Y)], axis=1)

# ----------------------------------- forward model -----------------------------------
t = np.arange(T_START, T_MAX, DT)
n = len(SENSOR_XS)


def pulse(tau):
    return np.exp(-(tau**2) / (2 * PULSE_WIDTH**2))


signals = np.stack(
    [
        born_forward(sensor_positions[e], sensor_positions, z, x, y, t, C, pulse)
        for e in range(n)
    ]
)

# ----------------------------------- reconstruction ----------------------------------
# delay-and-sum: remove the direct arrival, then backproject over all emitters
signals_das = signals.copy()
for e in range(n):
    for j, sensor_pos in enumerate(sensor_positions):
        tof_direct = np.linalg.norm(sensor_pos - sensor_positions[e]) / C
        signals_das[e, j] -= pulse(t - tof_direct)

das = np.zeros((NX, NY))
for e in range(n):
    das += das_backproject(
        sensor_positions[e], sensor_positions, signals_das[e], x, y, t, C
    )

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(NX / 10, NY / 10), dpi=100)
ax.imshow(z.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()

fig, ax = plt.subplots(figsize=(NX / 10, NY / 10), dpi=100)
ax.imshow(das.T, origin="lower", cmap="binary")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
