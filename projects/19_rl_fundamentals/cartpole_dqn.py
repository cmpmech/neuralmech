import argparse
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from NN import MLP
from helper import MAX_STEPS, reset_cartpole, step_cartpole
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
LR = 5e-4
GAMMA = 0.99  # discount factor
BATCH_SIZE = 64
BUFFER = 50_000  # replay buffer capacity in transitions
EPSILON_STEPS = 10_000  # environment steps over which exploration decays from 1
EPSILON_MIN = 0.01
TARGET_UPDATE = 500  # environment steps between copies to the target network
SOLVED = 475  # mean return over the last WINDOW episodes that counts as solved
WINDOW = 20

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [4, 128, 128, 2]  # state -> action values Q(s, left), Q(s, right)
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
target_model = copy.deepcopy(model)
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
buffer_states = torch.zeros(BUFFER, 4)
buffer_actions = torch.zeros(BUFFER, dtype=torch.long)
buffer_next_states = torch.zeros(BUFFER, 4)
buffer_fallen = torch.zeros(BUFFER)

episode_return = []
step = 0
tic = time.time()
pbar = tqdm(range(EPISODES))
for episode in pbar:
    state = reset_cartpole(rng)
    for tau in range(MAX_STEPS):
        epsilon = max(EPSILON_MIN, 1 - step / EPSILON_STEPS)
        if rng.random() < epsilon:
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                q = model(torch.as_tensor(state, dtype=torch.float32).to(device))
            action = q.argmax().item()
        next_state, fallen = step_cartpole(state, action)

        idx = step % BUFFER
        buffer_states[idx] = torch.as_tensor(state)
        buffer_actions[idx] = action
        buffer_next_states[idx] = torch.as_tensor(next_state)
        buffer_fallen[idx] = float(fallen)
        state = next_state
        step += 1

        if step >= BATCH_SIZE:
            ids = torch.as_tensor(rng.integers(min(step, BUFFER), size=BATCH_SIZE))
            s = buffer_states[ids].to(device)
            a = buffer_actions[ids].to(device)
            s_next = buffer_next_states[ids].to(device)
            not_fallen = 1 - buffer_fallen[ids].to(device)
            with torch.no_grad():
                q_next = target_model(s_next).max(dim=1).values
            target = 1.0 + GAMMA * not_fallen * q_next
            q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)

            optimizer.zero_grad()
            cost = cost_fun(q, target)
            cost.backward()
            optimizer.step()

        if step % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())
        if fallen:
            break

    episode_return.append(tau + 1)
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
        CSV_DIR / "cartpole_dqn.csv",
        episode=np.arange(len(episode_return)),
        steps=steps,
        episode_return=episode_return,
        mean_return=mean_return,
    )
