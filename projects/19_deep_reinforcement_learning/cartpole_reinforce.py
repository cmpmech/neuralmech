import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small
rng = np.random.default_rng(0)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPISODES = 2000  # upper bound, training stops once solved
LR = 1e-2
GAMMA = 0.99  # discount factor
SOLVED = 475  # mean return over the last WINDOW episodes that counts as solved
WINDOW = 20

# physics
GRAVITY = 9.8  # m/s^2
CART_MASS = 1.0  # kg
POLE_MASS = 0.1  # kg
POLE_LENGTH = 0.5  # m, half the pole length
FORCE = 10.0  # N, magnitude of a push
DT = 0.02  # s, explicit euler time step
MAX_STEPS = 500  # episode cap, also the maximum return
X_LIMIT = 2.4  # m
THETA_LIMIT = 12 * np.pi / 180  # rad

# model settings
LAYERS = [4, 128, 2]  # state (x, x_dot, theta, theta_dot) -> logits (left, right)
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]

RESOLUTION = 100  # grid of the policy map


# --------------------------------------- helper --------------------------------------
def step_cartpole(state, action):
    x, x_dot, theta, theta_dot = state
    force = FORCE if action == 1 else -FORCE
    total_mass = CART_MASS + POLE_MASS
    sin, cos = np.sin(theta), np.cos(theta)
    temp = (force + POLE_MASS * POLE_LENGTH * theta_dot**2 * sin) / total_mass
    theta_acc = (GRAVITY * sin - cos * temp) / (
        POLE_LENGTH * (4 / 3 - POLE_MASS * cos**2 / total_mass)
    )
    x_acc = temp - POLE_MASS * POLE_LENGTH * theta_acc * cos / total_mass
    state = state + DT * np.array([x_dot, x_acc, theta_dot, theta_acc])
    done = abs(state[0]) > X_LIMIT or abs(state[2]) > THETA_LIMIT
    return state, done


def run_episode(model, state, greedy=False):
    states = []
    log_probs = []
    for _ in range(MAX_STEPS):
        logits = model(torch.as_tensor(state, dtype=torch.float32).to(device))
        policy = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if greedy else policy.sample()
        states.append(state)
        log_probs.append(policy.log_prob(action))
        state, done = step_cartpole(state, action.item())
        if done:
            break
    return np.array(states), torch.stack(log_probs)


def compute_returns(steps):
    returns = np.zeros(steps)
    g = 0.0
    for t in reversed(range(steps)):
        g = 1.0 + GAMMA * g  # reward is +1 for every step the pole stays up
        returns[t] = g
    return returns


def draw_cartpole(ax, state):
    x, theta = state[0], state[2]
    ax.clear()
    ax.plot([-X_LIMIT, X_LIMIT], [0, 0], "k", lw=1)
    ax.add_patch(plt.Rectangle((x - 0.25, -0.1), 0.5, 0.2, color="k"))
    tip_x = x + 2 * POLE_LENGTH * np.sin(theta)
    tip_y = 2 * POLE_LENGTH * np.cos(theta)
    ax.plot([x, tip_x], [0, tip_y], "r", lw=4)
    ax.set_xlim(-X_LIMIT, X_LIMIT)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------- instantiate model & optimizer ---------------------------
modules = []
for i in range(len(LAYERS) - 2):
    modules += [nn.Linear(LAYERS[i], LAYERS[i + 1]), ACTIVATIONS[i]]
modules.append(nn.Linear(LAYERS[-2], LAYERS[-1]))
model = nn.Sequential(*modules)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
state_test = rng.uniform(-0.05, 0.05, 4)
model.eval()
with torch.no_grad():
    states_untrained, _ = run_episode(model, state_test, greedy=True)

episode_return = []
tic = time.time()
print_every = 10
pbar = tqdm(range(EPISODES))
for episode in pbar:
    model.train()
    states, log_probs = run_episode(model, rng.uniform(-0.05, 0.05, 4))
    returns = torch.tensor(compute_returns(len(states)), dtype=torch.float32)
    returns = returns.to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # acts as a baseline
    optimizer.zero_grad()
    cost = -(log_probs * returns).sum()
    cost.backward()
    optimizer.step()
    episode_return.append(len(states))

    mean_return = np.mean(episode_return[-WINDOW:])
    if episode % print_every == 0:
        pbar.set_postfix({"return": len(states), "mean": f"{mean_return:.1f}"})
    if len(episode_return) >= WINDOW and mean_return >= SOLVED:
        print(f"solved at episode {episode} (mean return {mean_return:.1f})")
        break
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
model.eval()
with torch.no_grad():
    states_trained, _ = run_episode(model, state_test, greedy=True)

    theta = torch.linspace(-THETA_LIMIT, THETA_LIMIT, RESOLUTION)
    theta_dot = torch.linspace(-2, 2, RESOLUTION)
    theta, theta_dot = torch.meshgrid(theta, theta_dot, indexing="ij")
    zeros = torch.zeros(RESOLUTION**2)
    x_test = torch.stack([zeros, zeros, theta.flatten(), theta_dot.flatten()], dim=1)
    push_right = torch.softmax(model(x_test.to(device)), dim=1)[:, 1]
    push_right = push_right.reshape(RESOLUTION, RESOLUTION).cpu().numpy()

episode_return = np.array(episode_return)
moving_average = np.convolve(episode_return, np.ones(WINDOW) / WINDOW, mode="valid")
time_untrained = np.arange(len(states_untrained)) * DT
time_trained = np.arange(len(states_trained)) * DT

fig_return, ax = plt.subplots()
ax.plot(episode_return, "k", lw=0.5)
ax.plot(np.arange(WINDOW - 1, len(episode_return)), moving_average, "r")
ax.set_xlabel("episode")
ax.set_ylabel("return (steps balanced)")

fig_trajectory, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(time_trained, np.degrees(states_trained[:, 2]), "k")
axs[0].plot(time_untrained, np.degrees(states_untrained[:, 2]), "r")
axs[0].axhline(np.degrees(THETA_LIMIT), color="b", ls="--")
axs[0].axhline(-np.degrees(THETA_LIMIT), color="b", ls="--")
axs[0].set_ylabel("pole angle [deg]")
axs[1].plot(time_trained, states_trained[:, 0], "k")
axs[1].plot(time_untrained, states_untrained[:, 0], "r")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("cart position [m]")

fig_policy, ax = plt.subplots()
extent = [-np.degrees(THETA_LIMIT), np.degrees(THETA_LIMIT), -2, 2]
im = ax.imshow(
    push_right.T, origin="lower", extent=extent, aspect="auto", cmap="cividis"
)
im.set_clim(0, 1)
ax.plot(np.degrees(states_trained[:, 2]), states_trained[:, 3], "r", lw=0.5)
ax.set_xlabel("pole angle [deg]")
ax.set_ylabel("pole angular velocity [rad/s]")
fig_policy.colorbar(im, ax=ax)

if not args.book:
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    for fig in [fig_return, fig_trajectory, fig_policy]:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_return.savefig(RGB_PDF_DIR / "cartpole_reinforce_return.pdf")
    fig_trajectory.savefig(RGB_PDF_DIR / "cartpole_reinforce_trajectory.pdf")
    fig_policy.savefig(RGB_PDF_DIR / "cartpole_reinforce_policy.pdf")
    save_csv(
        CSV_DIR / "cartpole_reinforce_return.csv",
        episode=np.arange(len(episode_return)),
        episode_return=episode_return,
    )
    save_csv(
        CSV_DIR / "cartpole_reinforce_trajectory.csv",
        t=time_trained,
        x=states_trained[:, 0],
        theta=states_trained[:, 2],
    )

# ----------------------------------- animate export ----------------------------------
if args.animate:
    ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/cartpole_reinforce"
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for f, state in enumerate(states_trained[::2]):
        draw_cartpole(ax, state)
        fig.savefig(ANIMATION_DIR / f"frame_{f:04d}.jpg")
    plt.close(fig)
