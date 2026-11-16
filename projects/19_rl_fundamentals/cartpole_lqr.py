import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

from helper import (
    CART_MASS,
    FORCE,
    GRAVITY,
    MAX_STEPS,
    POLE_LENGTH,
    POLE_MASS,
    reset_cartpole,
    step_cartpole,
)
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPISODES = 100
STATE_COST = np.diag([1.0, 1.0, 10.0, 1.0])  # weights of x, x_dot, theta, theta_dot
FORCE_COST = 1.0
STEPS_MAX = 200_000  # right end of learning-curve axis

# ---------------------------------- create controller --------------------------------
# equations of motion linearized about upright rest state
total_mass = CART_MASS + POLE_MASS
denominator = POLE_LENGTH * (4 / 3 - POLE_MASS / total_mass)
theta_acc_theta = GRAVITY / denominator
theta_acc_force = -1 / (total_mass * denominator)
x_acc_theta = -POLE_MASS * POLE_LENGTH * theta_acc_theta / total_mass
x_acc_force = 1 / total_mass - POLE_MASS * POLE_LENGTH * theta_acc_force / total_mass
A = np.array(
    [[0, 1, 0, 0], [0, 0, x_acc_theta, 0], [0, 0, 0, 1], [0, 0, theta_acc_theta, 0]]
)
B = np.array([[0], [x_acc_force], [0], [theta_acc_force]])
P = solve_continuous_are(A, B, STATE_COST, np.array([[FORCE_COST]]))
K = (B.T @ P) / FORCE_COST

# --------------------------------------- rollout -------------------------------------
episode_return = []
for episode in range(EPISODES):
    state = reset_cartpole(rng)
    for step in range(MAX_STEPS):
        action = int(-(K @ state).item() > 0)  # push right if linear force is positive
        state, fallen = step_cartpole(state, action)
        if fallen:
            break
    episode_return.append(step + 1)  # every step counts, as in other drivers
mean_return = np.mean(episode_return)
print(f"mean return {mean_return:.1f}, minimum {min(episode_return)}")

if not args.book:
    plt.hist(episode_return)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "cartpole_lqr.csv",
        steps=[0, STEPS_MAX],
        mean_return=[mean_return, mean_return],
    )
