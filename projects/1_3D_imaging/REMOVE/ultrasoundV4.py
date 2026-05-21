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
DT = 0.01
T_START = -0.3
T_MAX = 3.0
PULSE_WIDTH = 0.04

# sensors / emitter (physical coordinates)
SENSOR_XS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
SENSOR_Y = 1.0
EMITTER_IDX = 0

# defect (physical coordinates)
DEFECT_X = 0.1
DEFECT_Y = 0.3

# material grid (resolution only)
NX = 10
NY = 10

# sector scan
N_THETA = 300
N_R = 300
R_MIN = 0.05
THETA_MAX = np.pi / 3

# -------------------- domain + reflectivity -------------------
x = np.linspace(0, 1, NX)
y = np.linspace(0, 1, NY)

z = np.zeros((NX, NY))
z[round(DEFECT_X * (NX - 1)), round(DEFECT_Y * (NY - 1))] = 1.0

# ---------------------- forward model -------------------------
emitter_pos = np.array([SENSOR_XS[EMITTER_IDX], SENSOR_Y])

t = np.arange(T_START, T_MAX, DT)
signals = np.zeros((len(SENSOR_XS), len(t)))


def pulse(tau):
    return np.exp(-(tau**2) / (2 * PULSE_WIDTH**2))


# source
for j, sx in enumerate(SENSOR_XS):
    tof_direct = np.linalg.norm(np.array([sx, SENSOR_Y]) - emitter_pos) / C
    signals[j] += pulse(t - tof_direct)

# reflection (Born approximation)
for ix in range(NX):
    for iy in range(NY):
        if z[ix, iy] == 0.0:
            continue
        cell = np.array([x[ix], y[iy]])
        d_emit = np.linalg.norm(cell - emitter_pos)
        for j, sx in enumerate(SENSOR_XS):
            tof = (d_emit + np.linalg.norm(cell - np.array([sx, SENSOR_Y]))) / C
            signals[j] += z[ix, iy] * pulse(t - tof)

# ---------------------- sector scan grid ----------------------
# apex at the emitter position; range covers the full signal window
R_MAX = (T_MAX - T_START) / C

theta = np.linspace(-THETA_MAX, THETA_MAX, N_THETA)
r = np.linspace(R_MIN, R_MAX, N_R)
r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")  # (N_R, N_THETA)

x_fan = emitter_pos[0] + r_grid * np.sin(theta_grid)
y_fan = emitter_pos[1] - r_grid * np.cos(theta_grid)

# ---------------------------- delay-and-sum -----------------------------
signals_das = signals.copy()
for j, sx in enumerate(SENSOR_XS):
    tof_direct = np.linalg.norm(np.array([sx, SENSOR_Y]) - emitter_pos) / C
    signals_das[j] -= pulse(t - tof_direct)

das = np.zeros((N_R, N_THETA))
for j, sx in enumerate(SENSOR_XS):
    d_emit = np.sqrt((x_fan - emitter_pos[0]) ** 2 + (y_fan - emitter_pos[1]) ** 2)
    d_recv = np.sqrt((x_fan - sx) ** 2 + (y_fan - SENSOR_Y) ** 2)
    tofs = (d_emit + d_recv) / C
    das += np.interp(tofs.ravel(), t, signals_das[j]).reshape(N_R, N_THETA)

# --------------------------- plot -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]
ax.pcolormesh(x, y, z.T, cmap="binary")
for i, sx in enumerate(SENSOR_XS):
    fc = "tab:red" if i == EMITTER_IDX else "tab:blue"
    ax.scatter(sx, SENSOR_Y, color=fc, zorder=5, s=50, clip_on=False)
ax.set_aspect("equal")

ax = axes[1]
vmax = np.max(np.abs(das)) or 1.0
ax.pcolormesh(x_fan, y_fan, das, cmap="seismic", vmin=-vmax, vmax=vmax, shading="gouraud")
for i, sx in enumerate(SENSOR_XS):
    fc = "tab:red" if i == EMITTER_IDX else "tab:blue"
    ax.scatter(sx, SENSOR_Y, color=fc, zorder=5, s=30, clip_on=False)
ax.set_facecolor("black")
ax.set_aspect("equal")

plt.tight_layout()

if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "ultrasoundV4.pdf")
    plt.close()
elif args.animate:
    plt.close()
else:
    plt.show()
