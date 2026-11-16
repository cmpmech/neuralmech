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

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 6000  # one freshly drawn batch of random moves per epoch
LR = 1e-3
BATCH_SIZE = 256
MOMENTUM = 0.9  # rate of the moving average of the target encoder

# model settings
D = 32  # embedding width
ENCODER = [2 * 25 * 25, 256, 256, D]
PREDICTOR = [D + 4, 256, 256, D]

# maze
MAZE = 12  # cells per side, the maze of the chapter 1 probes

# --------------------------------------- helper --------------------------------------
rng = np.random.default_rng(0)
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right


def generate_maze(n):
    walls = np.ones((2 * n + 1, 2 * n + 1), dtype=bool)
    stack = [(0, 0)]
    visited = {(0, 0)}
    walls[1, 1] = False
    while stack:
        i, j = stack[-1]
        neighbors = [(i + di, j + dj) for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0))
                     if 0 <= i + di < n and 0 <= j + dj < n and (i + di, j + dj) not in visited]
        if not neighbors:
            stack.pop()
            continue
        k, m = neighbors[rng.integers(len(neighbors))]
        walls[i + k + 1, j + m + 1] = False
        walls[2 * k + 1, 2 * m + 1] = False
        visited.add((k, m))
        stack.append((k, m))
    return walls


def step(cells, actions):
    move = MOVES[actions]
    blocked = walls[2 * cells[:, 0] + 1 + move[:, 0], 2 * cells[:, 1] + 1 + move[:, 1]]
    return cells + move * ~blocked[:, None]


def observe(cells):
    images = np.zeros((len(cells), 2, 2 * MAZE + 1, 2 * MAZE + 1), dtype=np.float32)
    images[:, 0] = walls
    images[np.arange(len(cells)), 1, 2 * cells[:, 0] + 1, 2 * cells[:, 1] + 1] = 1
    return torch.tensor(images.reshape(len(cells), -1), device=device)


def predict(z, actions):
    a = nn.functional.one_hot(torch.as_tensor(actions, device=device), 4).float()
    return predictor(torch.cat([z, a], dim=-1))


def breadth_first_search(successors, start, goal):
    parents = {start: None}
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node == goal:
            break
        for a in range(4):
            nxt = successors[a, node]
            if nxt >= 0 and nxt not in parents:
                parents[nxt] = (node, a)
                queue.append(nxt)
    actions = []
    while parents.get(goal) is not None:
        goal, a = parents[goal]
        actions.append(a)
    return actions[::-1]


# ------------------------------------- create data -----------------------------------
generate_maze(6)  # the chapter 1 probes draw the 6 x 6 maze first from the same generator
walls = generate_maze(MAZE)
cells = np.stack(np.meshgrid(np.arange(MAZE), np.arange(MAZE), indexing="ij"), -1).reshape(-1, 2)
start, goal = 0, MAZE**2 - 1

# ---------------------------- instantiate model & optimizer --------------------------
encoder = MLP(ENCODER, [nn.GELU() for _ in range(len(ENCODER) - 2)]).to(device)
predictor = MLP(PREDICTOR, [nn.GELU() for _ in range(len(PREDICTOR) - 2)]).to(device)
target_encoder = copy.deepcopy(encoder).requires_grad_(False)
optimizer = torch.optim.Adam([*encoder.parameters(), *predictor.parameters()], lr=LR)

# -------------------------------------- training -------------------------------------
cost_history = []
tic = time.perf_counter()
for epoch in (pbar := tqdm(range(EPOCHS))):
    batch = rng.integers(MAZE, size=(BATCH_SIZE, 2))
    actions = rng.integers(4, size=BATCH_SIZE)
    z = encoder(observe(batch))
    z_target = target_encoder(observe(step(batch, actions)))
    covariance = torch.cov(z.T)
    variance = torch.relu(1 - covariance.diagonal().sqrt()).mean()
    decorrelation = (covariance**2).sum() / D - (covariance.diagonal() ** 2).sum() / D
    prediction = ((predict(z, actions) - z_target) ** 2).mean()
    cost = prediction + variance + decorrelation / 25  # weights 25, 25, 1 of VICReg
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    with torch.no_grad():
        for xi, theta in zip(target_encoder.parameters(), encoder.parameters()):
            xi.mul_(MOMENTUM).add_((1 - MOMENTUM) * theta)
    cost_history.append(prediction.item())
    pbar.set_postfix(cost=f"{prediction.item():.2e}")
toc = time.perf_counter()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
with torch.no_grad():
    z_cells = encoder(observe(cells))
    z_next = torch.stack([predict(z_cells, [a] * len(cells)) for a in range(4)])
    separation = torch.cdist(z_cells, z_cells).fill_diagonal_(torch.inf).min().item()
    distances = torch.cdist(z_next, z_cells[None].expand(4, -1, -1))
    successors = distances.argmin(-1).cpu().numpy()
    successors[distances.min(-1).values.cpu().numpy() > separation / 2] = -1  # no state matches
true_successors = np.stack([step(cells, np.full(len(cells), a)) @ [MAZE, 1] for a in range(4)])
print(f"states {separation:.2f} apart, {(successors == true_successors).mean():.2%} of the moves predicted correctly")

actions = breadth_first_search(successors, start, goal)
path = [cells[start]]
for a in actions:
    path.append(step(path[-1][None], np.array([a]))[0])
path = np.array(path)
shortest = len(breadth_first_search(true_successors, start, goal))
print(f"planned {len(actions)} moves (shortest {shortest}), goal reached: {(path[-1] == cells[goal]).all()}")

fig_moves, ax = plt.subplots(figsize=(4, 4))
ax.imshow(walls, cmap="binary", vmax=2.5, interpolation="nearest")
for a in range(4):
    for i, j in enumerate(successors[a]):
        if j >= 0 and j != i:
            color = "tab:blue" if j == true_successors[a, i] else "tab:red"
            ax.plot(2 * cells[[i, j], 1] + 1, 2 * cells[[i, j], 0] + 1, color=color, linewidth=1.5)
ax.axis("off")
fig_moves.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig_maze, ax = plt.subplots(figsize=(4, 4))
ax.imshow(walls, cmap="binary", interpolation="nearest")
ax.plot(2 * path[:, 1] + 1, 2 * path[:, 0] + 1, "tab:blue", linewidth=2)
ax.plot(1, 1, "o", color="tab:green", markersize=200 / MAZE)
ax.plot(2 * MAZE - 1, 2 * MAZE - 1, "o", color="tab:red", markersize=200 / MAZE)
ax.axis("off")
fig_maze.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.figure()
    plt.semilogy(cost_history)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig_maze.savefig(RGB_PDF_DIR / "jepa_maze_path.pdf")
    fig_moves.savefig(RGB_PDF_DIR / "jepa_maze_moves.pdf")
