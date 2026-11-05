import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from NN import MLP
from helper import DT, MAX_STEPS, phase_plane, reset_cartpole, step_cartpole
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small
torch.set_num_threads(1)  # threads only add overhead here, and change the rounding
rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPISODES = 3000  # upper bound, training stops once solved
LR = 1e-3
LR_VALUE = 1e-2  # critic, larger so it keeps up with the policy
GAMMA = 0.99  # discount factor
SOLVED = 475  # mean return over the last WINDOW episodes that counts as solved
WINDOW = 20

# model settings
LAYERS = [4, 128, 2]  # state (x, x_dot, theta, theta_dot) -> logits (left, right)
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]
LAYERS_VALUE = [4, 128, 1]  # state -> expected return
ACTIVATIONS_VALUE = [nn.ReLU() for _ in range(len(LAYERS_VALUE) - 2)]

RESOLUTION = 100  # phase-plane grid of the policy and value maps


# --------------------------------------- helper --------------------------------------
def run_episode(model, state, greedy=False):
    states = []
    log_probs = []
    for _ in range(MAX_STEPS):
        logits = model(torch.as_tensor(state, dtype=torch.float32).to(device))
        policy = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if greedy else policy.sample()
        states.append(state)
        log_probs.append(policy.log_prob(action))
        state, fallen = step_cartpole(state, action.item())
        if fallen:
            break
    states.append(state)
    return np.array(states), torch.stack(log_probs), fallen


# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), LR)

value_model = MLP(LAYERS_VALUE, ACTIVATIONS_VALUE)
value_model.to(device)
value_optimizer = torch.optim.Adam(value_model.parameters(), LR_VALUE)

# -------------------------------------- training -------------------------------------
state_test = np.array([0.0, 0.0, 0.05, 0.0])  # pole tilted by 3 degrees
with torch.no_grad():
    states_untrained, _, _ = run_episode(model, state_test, greedy=True)

episode_return = []
tic = time.time()
pbar = tqdm(range(EPISODES))
for episode in pbar:
    states, log_probs, fallen = run_episode(model, reset_cartpole(rng))
    rewards = torch.ones(len(log_probs), device=device)
    values = value_model(torch.tensor(states, dtype=torch.float32).to(device))
    values = values.squeeze(1)
    not_fallen = torch.ones(len(log_probs), device=device)
    not_fallen[-1] = 0.0 if fallen else 1.0  # step cap only truncates, bootstrap
    targets = rewards + GAMMA * not_fallen * values[1:].detach()
    td_errors = targets - values[:-1]

    optimizer.zero_grad()
    advantages = td_errors.detach()
    cost = -(log_probs * advantages).sum()
    cost.backward()
    optimizer.step()

    value_optimizer.zero_grad()
    cost_value = torch.mean(td_errors**2)
    cost_value.backward()
    value_optimizer.step()

    episode_return.append(len(rewards))
    mean_return = np.mean(episode_return[-WINDOW:])
    pbar.set_postfix({"return": episode_return[-1], "mean": f"{mean_return:.1f}"})
    if len(episode_return) >= WINDOW and mean_return >= SOLVED:
        break
toc = time.time()
print(f"{episode + 1} episodes, {int(np.sum(episode_return))} environment steps")
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
episode_return = np.array(episode_return)
steps = np.cumsum(episode_return)
mean_return = [
    episode_return[max(0, i + 1 - WINDOW) : i + 1].mean()
    for i in range(len(episode_return))
]

x_test, extent = phase_plane(RESOLUTION)
x_test = x_test.to(device)
with torch.no_grad():
    states_trained, _, _ = run_episode(model, state_test, greedy=True)
    push_right = torch.softmax(model(x_test), dim=1)[:, 1]
    push_right = push_right.reshape(RESOLUTION, RESOLUTION).cpu().numpy()
    value = value_model(x_test).reshape(RESOLUTION, RESOLUTION).cpu().numpy()
time_untrained = np.arange(len(states_untrained)) * DT
time_trained = np.arange(len(states_trained)) * DT

fig_policy, ax = plt.subplots()
ax.imshow(push_right.T, origin="lower", extent=extent, aspect="auto", cmap="cividis")
fig_value, ax = plt.subplots()
ax.imshow(value.T, origin="lower", extent=extent, aspect="auto", cmap="cividis")

if not args.book:
    for fig in [fig_policy, fig_value]:
        fig.axes[0].plot(np.degrees(states_trained[:, 2]), states_trained[:, 3], "r")
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(steps, episode_return, "k")
    axs[0].set_xlabel("environment steps")
    axs[0].set_ylabel("return")
    axs[1].plot(time_trained, np.degrees(states_trained[:, 2]), "k")
    axs[1].plot(time_untrained, np.degrees(states_untrained[:, 2]), "r")
    axs[1].set_ylabel("pole angle [deg]")
    axs[2].plot(time_trained, states_trained[:, 0], "k")
    axs[2].plot(time_untrained, states_untrained[:, 0], "r")
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("cart position [m]")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "cartpole_actor_critic.csv",
        episode=np.arange(len(episode_return)),
        steps=steps,
        episode_return=episode_return,
        mean_return=mean_return,
    )
    save_csv(
        CSV_DIR / "cartpole_actor_critic_trained.csv",
        t=time_trained,
        x=states_trained[:, 0],
        theta=np.degrees(states_trained[:, 2]),
        theta_dot=states_trained[:, 3],
    )
    save_csv(
        CSV_DIR / "cartpole_actor_critic_untrained.csv",
        t=time_untrained,
        x=states_untrained[:, 0],
        theta=np.degrees(states_untrained[:, 2]),
        theta_dot=states_untrained[:, 3],
    )
    for fig in [fig_policy, fig_value]:
        fig.axes[0].axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_policy.savefig(RGB_PDF_DIR / "cartpole_actor_critic_policy.pdf")
    fig_value.savefig(RGB_PDF_DIR / "cartpole_actor_critic_value.pdf")
    print(f"value map range {value.min():.2f} to {value.max():.2f}")
