import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
GAMMA = 0.99  # discount factor per time step, horizon of 1 / (1 - GAMMA) steps
TOLERANCE = 1e-6  # largest change of the value function at convergence
ITERS = 10_000  # upper bound

# physics
GRAVITY = 9.81  # m/s^2
MASS = 1.0  # kg, point mass at the tip
LENGTH = 1.0  # m
TORQUES = [-3.0, 0.0, 3.0]  # Nm, actions, too weak to lift the mass directly
DT = 0.05  # s, semi-implicit euler time step
OMEGA_MAX = 8.0  # rad/s, range of the angular velocity grid

# discretization
RESOLUTION = 201
T = 200  # time steps of the swing-up trajectory


# --------------------------------------- helper --------------------------------------
def step_pendulum(theta, omega, torque):
    alpha = GRAVITY / LENGTH * np.sin(theta) + torque / (MASS * LENGTH**2)
    omega = np.clip(omega + DT * alpha, -OMEGA_MAX, OMEGA_MAX)
    theta = (theta + DT * omega + np.pi) % (2 * np.pi) - np.pi
    return theta, omega


def interpolate(V, theta, omega):
    u = (theta + np.pi) / (2 * np.pi) * RESOLUTION
    v = (omega + OMEGA_MAX) / (2 * OMEGA_MAX) * (RESOLUTION - 1)
    i, j = np.floor(u).astype(int), np.minimum(np.floor(v).astype(int), RESOLUTION - 2)
    wu, wv = u - i, v - j
    i0, i1 = i % RESOLUTION, (i + 1) % RESOLUTION  # theta is periodic
    return (
        (1 - wu) * (1 - wv) * V[i0, j]
        + wu * (1 - wv) * V[i1, j]
        + (1 - wu) * wv * V[i0, j + 1]
        + wu * wv * V[i1, j + 1]
    )


# ------------------------------------ create data ------------------------------------
theta = np.linspace(-np.pi, np.pi, RESOLUTION, endpoint=False)
omega = np.linspace(-OMEGA_MAX, OMEGA_MAX, RESOLUTION)
theta, omega = np.meshgrid(theta, omega, indexing="ij")
reward = np.cos(theta)  # height of the mass over the pivot, in units of LENGTH
next_states = [step_pendulum(theta, omega, torque) for torque in TORQUES]

# ---------------------------------- value iteration ----------------------------------
V = np.zeros((RESOLUTION, RESOLUTION))
tic = time.time()
for k in range(ITERS):
    Q = np.stack([reward + GAMMA * interpolate(V, *s) for s in next_states])
    V_new = Q.max(axis=0)
    change = np.abs(V_new - V).max()
    V = V_new
    if change < TOLERANCE:
        break
toc = time.time()
policy = Q.argmax(axis=0)
print(f"converged after {k + 1} iterations")
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
trajectory = [(np.pi - 1e-3, 0.0)]  # hanging down, at rest
for tau in range(T):
    th, om = trajectory[-1]
    q = [GAMMA * interpolate(V, *step_pendulum(th, om, torque)) for torque in TORQUES]
    trajectory.append(step_pendulum(th, om, TORQUES[int(np.argmax(q))]))
trajectory = np.array(trajectory)

extent = [-180, 180, -OMEGA_MAX, OMEGA_MAX]
fig_value, ax = plt.subplots()
ax.imshow(V.T, origin="lower", extent=extent, aspect="auto", cmap="cividis")
fig_policy, ax = plt.subplots()
ax.imshow(policy.T, origin="lower", extent=extent, aspect="auto", cmap="cividis")

if not args.book:
    for fig in [fig_value, fig_policy]:
        ax = fig.axes[0]
        ax.plot(np.degrees(trajectory[:, 0]), trajectory[:, 1], "r.", ms=2)
        ax.set_xlabel("pole angle [deg]")
        ax.set_ylabel("angular velocity [rad/s]")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for fig in [fig_value, fig_policy]:
        fig.axes[0].axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_value.savefig(RGB_PDF_DIR / "pendulum_value_iteration_value.pdf")
    fig_policy.savefig(RGB_PDF_DIR / "pendulum_value_iteration_policy.pdf")
    theta_deg = np.degrees(trajectory[:, 0])
    theta_deg[1:][np.abs(np.diff(theta_deg)) > 180] = np.inf  # break line at wrap
    save_csv(
        CSV_DIR / "pendulum_value_iteration.csv",
        t=np.arange(T + 1) * DT,
        theta=theta_deg,
        omega=trajectory[:, 1],
    )
    print(f"value range {V.min():.2f} to {V.max():.2f}")
