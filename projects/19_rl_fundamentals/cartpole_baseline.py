import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from NN import MLP
from helper import MAX_STEPS, discounted_returns, reset_cartpole, step_cartpole
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
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
LR_VALUE = 1e-2  # critic, larger so the baseline keeps up with the policy
GAMMA = 0.99  # discount factor
SOLVED = 475  # mean return over the last WINDOW episodes that counts as solved
WINDOW = 20

# model settings
LAYERS = [4, 128, 2]  # state (x, x_dot, theta, theta_dot) -> logits (left, right)
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]
LAYERS_VALUE = [4, 128, 1]  # state -> expected return
ACTIVATIONS_VALUE = [nn.ReLU() for _ in range(len(LAYERS_VALUE) - 2)]


# --------------------------------------- helper --------------------------------------
def run_episode(model, state):
    states = []
    log_probs = []
    for _ in range(MAX_STEPS):
        logits = model(torch.as_tensor(state, dtype=torch.float32).to(device))
        policy = torch.distributions.Categorical(logits=logits)
        action = policy.sample()
        states.append(state)
        log_probs.append(policy.log_prob(action))
        state, fallen = step_cartpole(state, action.item())
        if fallen:
            break
    return np.array(states), torch.stack(log_probs)


# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), LR)

value_model = MLP(LAYERS_VALUE, ACTIVATIONS_VALUE)
value_model.to(device)
value_optimizer = torch.optim.Adam(value_model.parameters(), LR_VALUE)

# -------------------------------------- training -------------------------------------
episode_return = []
tic = time.time()
pbar = tqdm(range(EPISODES))
for episode in pbar:
    states, log_probs = run_episode(model, reset_cartpole(rng))
    rewards = np.ones(len(log_probs))
    returns = torch.tensor(discounted_returns(rewards, GAMMA), dtype=torch.float32)
    returns = returns.to(device)
    values = value_model(torch.tensor(states, dtype=torch.float32).to(device))
    values = values.squeeze(1)

    optimizer.zero_grad()
    advantages = (returns - values).detach()
    cost = -(log_probs * advantages).sum()
    cost.backward()
    optimizer.step()

    value_optimizer.zero_grad()
    cost_value = torch.mean((returns - values) ** 2)
    cost_value.backward()
    value_optimizer.step()

    episode_return.append(rewards.sum())
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

if not args.book:
    plt.plot(steps, episode_return, "k")
    plt.xlabel("environment steps")
    plt.ylabel("return")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "cartpole_baseline.csv",
        episode=np.arange(len(episode_return)),
        steps=steps,
        episode_return=episode_return,
        mean_return=mean_return,
    )
