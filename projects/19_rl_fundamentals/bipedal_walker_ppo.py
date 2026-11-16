import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small
torch.set_num_threads(1)  # threads only add overhead here, and change the rounding

# -------------------------------------- settings -------------------------------------
# hyperparameters
ITERS = 1200  # upper bound, training stops once solved
ENVS = 16  # environments simulated side by side
STEPS_PER_ENV = 256  # steps collected per environment with one policy
EPOCHS = 10  # passes over each batch of collected steps
BATCH_SIZE = 128
LR = 3e-4
GAMMA = 0.99  # discount factor
LAMBDA = 0.95  # generalized advantage estimation, 1 is the Monte Carlo return
CLIP = 0.2  # allowed change of the probability ratio
MAX_NORM = 0.5  # gradient clipping
SOLVED = 300  # mean return over the last WINDOW episodes that counts as solved
WINDOW = 100
SAVE_EVERY = 25  # iterations between saved policies

# model settings
LAYERS = [24, 128, 128, 4]  # state -> mean of the four joint torques
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]
LAYERS_VALUE = [24, 128, 128, 1]  # state -> expected return
ACTIVATIONS_VALUE = [nn.Tanh() for _ in range(len(LAYERS_VALUE) - 2)]


# --------------------------------------- helper --------------------------------------
def gaussian_policy(states):
    return torch.distributions.Normal(model(states), model.log_std.exp())


# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)
model.log_std = nn.Parameter(torch.zeros(LAYERS[-1], device=device))  # one per joint
optimizer = torch.optim.Adam(model.parameters(), LR)

value_model = MLP(LAYERS_VALUE, ACTIVATIONS_VALUE)
value_model.to(device)
value_optimizer = torch.optim.Adam(value_model.parameters(), LR)

# --------------------------------------- setup ---------------------------------------
envs = gym.make_vec(
    "BipedalWalker-v3", num_envs=ENVS, vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
)
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
envs = gym.wrappers.vector.NormalizeReward(envs, gamma=GAMMA)  # falls cost -100
state, _ = envs.reset(seed=0)

# -------------------------------------- training -------------------------------------
states = torch.zeros(STEPS_PER_ENV, ENVS, LAYERS[0], device=device)
actions = torch.zeros(STEPS_PER_ENV, ENVS, LAYERS[-1], device=device)
rewards = torch.zeros(STEPS_PER_ENV, ENVS, device=device)
dones = torch.zeros(STEPS_PER_ENV, ENVS, device=device)
episode_return = []
episode_steps = []
tic = time.time()
pbar = tqdm(range(ITERS))
for iteration in pbar:
    if iteration % SAVE_EVERY == 0:
        torch.save(model, MODEL_DIR / f"bipedal_walker_ppo_{iteration}.pt")
    with torch.no_grad():
        for tau in range(STEPS_PER_ENV):
            states[tau] = torch.as_tensor(state, dtype=torch.float32).to(device)
            actions[tau] = gaussian_policy(states[tau]).sample()
            state, reward, terminated, truncated, info = envs.step(
                actions[tau].clamp(-1, 1).cpu().numpy()
            )
            for idx in np.flatnonzero(truncated):  # time limit, the walk goes on
                final_state = torch.tensor(info["final_obs"][idx], dtype=torch.float32)
                reward[idx] += GAMMA * value_model(final_state.to(device)).item()
            rewards[tau] = torch.as_tensor(reward, dtype=torch.float32).to(device)
            dones[tau] = torch.as_tensor(terminated | truncated).to(device)
            if "episode" in info:
                episode_return += list(info["episode"]["r"][info["_episode"]])
                episode_steps += list(info["episode"]["l"][info["_episode"]])

        values = value_model(states).squeeze(-1)
        last_state = torch.as_tensor(state, dtype=torch.float32).to(device)
        next_values = torch.cat([values[1:], value_model(last_state).T])
        deltas = rewards + GAMMA * (1 - dones) * next_values - values
        advantages = torch.zeros_like(rewards)
        A = torch.zeros(ENVS, device=device)
        for tau in reversed(range(STEPS_PER_ENV)):
            A = deltas[tau] + GAMMA * LAMBDA * (1 - dones[tau]) * A
            advantages[tau] = A
        returns = advantages + values
        log_probs_old = gaussian_policy(states).log_prob(actions).sum(-1)

    batch = [
        x.flatten(0, 1) for x in (states, actions, advantages, returns, log_probs_old)
    ]
    for epoch in range(EPOCHS):
        for ids in torch.randperm(ENVS * STEPS_PER_ENV).split(BATCH_SIZE):
            s, a, A, G, log_prob_old = [x[ids] for x in batch]
            A = (A - A.mean()) / (A.std() + 1e-8)
            log_prob = gaussian_policy(s).log_prob(a).sum(-1)
            ratio = torch.exp(log_prob - log_prob_old)
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
            optimizer.zero_grad()
            cost = -torch.min(ratio * A, clipped * A).mean()
            cost.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

            value_optimizer.zero_grad()
            cost_value = torch.mean((G - value_model(s).squeeze(1)) ** 2)
            cost_value.backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), MAX_NORM)
            value_optimizer.step()

    mean_return = np.mean(episode_return[-WINDOW:]) if episode_return else np.nan
    pbar.set_postfix({"episodes": len(episode_return), "mean": f"{mean_return:.1f}"})
    if mean_return >= SOLVED:
        break
torch.save(model, MODEL_DIR / f"bipedal_walker_ppo_{iteration + 1}.pt")
toc = time.time()
print(f"{len(episode_return)} episodes, {(iteration + 1) * ENVS * STEPS_PER_ENV} steps")
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
episode_return = np.array(episode_return)
steps = np.cumsum(episode_steps)
mean_return = [
    episode_return[max(0, i + 1 - WINDOW) : i + 1].mean()
    for i in range(len(episode_return))
]

if not args.book:
    plt.plot(steps, episode_return, "k", alpha=0.3)
    plt.plot(steps, mean_return, "k")
    plt.xlabel("environment steps")
    plt.ylabel("return")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "bipedal_walker_ppo.csv",
        episode=np.arange(len(episode_return)),
        steps=steps,
        episode_return=episode_return,
        mean_return=mean_return,
    )
