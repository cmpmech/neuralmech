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
NX = 10
NY = 10
C = 1.0
DT = 0.01
T_START = -0.3
T_MAX = 3.0
PULSE_WIDTH = 0.04
SENSOR_X_IDX = [2, 3, 4, 5, 6, 7]
SENSOR_Y = 0.0

# -------------------- domain + reflectivity -------------------
x = np.linspace(0, 1, NX)
y = np.linspace(0, 1, NY)

z = np.zeros((NX, NY))
z[4, 6] = 1.0

# --------------------- sensor geometry ------------------------
sensor_xs = x[SENSOR_X_IDX]
n_sensors = len(SENSOR_X_IDX)

# ---------------------- forward model -------------------------
t = np.arange(T_START, T_MAX, DT)
signals = np.zeros((n_sensors, n_sensors, len(t)))


def pulse(tau):
    return np.exp(-(tau**2) / (2 * PULSE_WIDTH**2))


xx, yy = np.meshgrid(x, y, indexing="ij")

for e, ex in enumerate(sensor_xs):
    emitter_pos = np.array([ex, SENSOR_Y])

    # source
    for j, sx in enumerate(sensor_xs):
        sensor_pos = np.array([sx, SENSOR_Y])
        tof_direct = np.linalg.norm(sensor_pos - emitter_pos) / C
        signals[e, j] += pulse(t - tof_direct)

    # reflection (Born approximation)
    for ix in range(NX):
        for iy in range(NY):
            if z[ix, iy] == 0.0:
                continue
            cell = np.array([x[ix], y[iy]])
            d_emit = np.linalg.norm(cell - emitter_pos)
            for j, sx in enumerate(sensor_xs):
                sensor_pos = np.array([sx, SENSOR_Y])
                tof = (d_emit + np.linalg.norm(cell - sensor_pos)) / C
                signals[e, j] += z[ix, iy] * pulse(t - tof)

# mute direct wave before reconstruction
for e, ex in enumerate(sensor_xs):
    emitter_pos = np.array([ex, SENSOR_Y])
    for j, sx in enumerate(sensor_xs):
        sensor_pos = np.array([sx, SENSOR_Y])
        tof_direct = np.linalg.norm(sensor_pos - emitter_pos) / C
        signals[e, j] -= pulse(t - tof_direct)

# ---------------------------- delay-and-sum -----------------------------
das = np.zeros((NX, NY))

for e, ex in enumerate(sensor_xs):
    d_emit = np.sqrt((xx - ex) ** 2 + yy ** 2)
    for j, sx in enumerate(sensor_xs):
        d_recv = np.sqrt((xx - sx) ** 2 + yy ** 2)
        tofs = (d_emit + d_recv) / C
        das += np.interp(tofs.ravel(), t, signals[e, j]).reshape(NX, NY)

# --------------------------- plot -----------------------------
compound = signals.sum(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.pcolormesh(x, y, z.T, cmap="binary")
for sx in sensor_xs:
    ax.scatter(sx, SENSOR_Y, color="tab:blue", zorder=5, s=50, clip_on=False)

ax = axes[1]
vmax_sig = np.max(np.abs(compound))
ax.imshow(
    compound,
    aspect="auto",
    extent=[T_START, T_MAX, n_sensors - 0.5, -0.5],
    cmap="seismic",
    vmin=-vmax_sig,
    vmax=vmax_sig,
)

ax = axes[2]
vmax_das = np.max(np.abs(das))
ax.pcolormesh(x, y, das.T, cmap="seismic", vmin=-vmax_das, vmax=vmax_das)

plt.tight_layout()

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "ultrasoundV3.pdf")
    plt.close()
elif args.animate:
    plt.close()
else:
    plt.show()
