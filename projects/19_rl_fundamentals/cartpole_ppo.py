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
ITERS = 300  # upper bound, training stops once solved
STEPS_PER_ITER = 2000  # environment steps collected with one policy
EPOCHS = 10  # passes over each batch of collected steps
BATCH_SIZE = 250
LR = 1e-3
LR_VALUE = 1e-2  # critic, larger so it keeps up with the policy
GAMMA = 0.99  # discount factor
CLIP = 0.2  # allowed change of the probability ratio
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
    actions = []
    for _ in range(MAX_STEPS):
        logits = model(torch.as_tensor(state, dtype=torch.float32).to(device))
        action = torch.distributions.Categorical(logits=logits).sample()
        states.append(state)
        actions.append(action.item())
        state, fallen = step_cartpole(state, action.item())
        if fallen:
            break
    return np.array(states), np.array(actions)


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
pbar = tqdm(range(ITERS))
for iteration in pbar:
    states, actions, returns = [], [], []
    with torch.no_grad():
        while sum(len(s) for s in states) < STEPS_PER_ITER:
            s, a = run_episode(model, reset_cartpole(rng))
            states.append(s)
            actions.append(a)
            returns.append(discounted_returns(np.ones(len(a)), GAMMA))
            episode_return.append(len(a))
    states = torch.tensor(np.concatenate(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.concatenate(actions)).to(device)
    returns = torch.tensor(np.concatenate(returns), dtype=torch.float32).to(device)
    with torch.no_grad():
        log_probs_old = torch.distributions.Categorical(logits=model(states))
        log_probs_old = log_probs_old.log_prob(actions)
        advantages = returns - value_model(states).squeeze(1)

    for epoch in range(EPOCHS):
        for ids in torch.randperm(len(actions)).split(BATCH_SIZE):
            policy = torch.distributions.Categorical(logits=model(states[ids]))
            ratio = torch.exp(policy.log_prob(actions[ids]) - log_probs_old[ids])
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
            optimizer.zero_grad()
            cost = -torch.min(ratio * advantages[ids], clipped * advantages[ids]).mean()
            cost.backward()
            optimizer.step()

            value_optimizer.zero_grad()
            values = value_model(states[ids]).squeeze(1)
            cost_value = torch.mean((returns[ids] - values) ** 2)
            cost_value.backward()
            value_optimizer.step()

    mean_return = np.mean(episode_return[-WINDOW:])
    pbar.set_postfix({"episodes": len(episode_return), "mean": f"{mean_return:.1f}"})
    if mean_return >= SOLVED:
        break
toc = time.time()
print(
    f"{len(episode_return)} episodes, {int(np.sum(episode_return))} environment steps"
)
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
        CSV_DIR / "cartpole_ppo.csv",
        episode=np.arange(len(episode_return)),
        steps=steps,
        episode_return=episode_return,
        mean_return=mean_return,
    )
