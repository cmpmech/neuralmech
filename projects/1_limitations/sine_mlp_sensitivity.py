import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------- training settings ---------------------------
EPOCHS = 800  # 600
REGULARIZATION = 0
BATCH_SIZE = 32

cost_fun = nn.MSELoss(reduction="mean")

# ---------------------------- model settings ----------------------------
# architecture fixed: only the init seed and learning rate vary
# layers = [1, 24, 24, 24, 1]
# layers = [1, 24, 24, 1]
layers = [1, 12, 12, 12, 1]
# activations = [nn.GELU(approximate="tanh")] * (len(layers) - 2)
activations = [nn.Tanh()] * (len(layers) - 2)
# activations = [nn.Sigmoid()] * (len(layers) - 2)


# --------------------------- sweep settings -----------------------------
SEED_COUNT = 100  # 80  # number of distinct init seeds
LR_RESOLUTION = 200  # 160
# LR_RANGE = (1e-4, 1e-1)
LR_RANGE = (1e-4, 1e0)

seed_axis = np.arange(SEED_COUNT)
lr_axis = np.logspace(np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1]), LR_RESOLUTION)

# ----------------------------- prepare data -----------------------------
data = np.load(DATA_DIR / "sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.float32),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)


# --------------------------- single training ----------------------------
def train_run(seed, lr):
    torch.manual_seed(seed)  # only the init + shuffling stochasticity changes
    model = MLP(layers, activations).to(device)
    init_weights(model, activations[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=REGULARIZATION)

    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = standardizex(x.to(device)), standardizey(y.to(device))
            optimizer.zero_grad()
            cost = cost_fun(model(x), y)
            cost.backward()
            optimizer.step()

    final_train = 0.0
    for x, y in train_loader:
        x, y = standardizex(x.to(device)), standardizey(y.to(device))
        final_train += cost_fun(model(x), y).item()
    final_train /= len(train_loader)

    final_val = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = standardizex(x.to(device)), standardizey(y.to(device))
            final_val += cost_fun(model(x), y).item()
    final_val /= len(val_loader)
    return final_train, final_val


# ------------------------------- sweep ----------------------------------
train_grid = np.zeros((len(lr_axis), len(seed_axis)))
val_grid = np.zeros((len(lr_axis), len(seed_axis)))
tic = time.time()
pbar = tqdm(total=len(lr_axis) * len(seed_axis))
for i, lr in enumerate(lr_axis):
    for j, seed in enumerate(seed_axis):
        train_grid[i, j], val_grid[i, j] = train_run(int(seed), float(lr))
        pbar.set_postfix(
            {"train": f"{train_grid[i, j]:.2e}", "val": f"{val_grid[i, j]:.2e}"}
        )
        pbar.update(1)
pbar.close()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

seed_mesh, lr_mesh = np.meshgrid(seed_axis, lr_axis)


lr, seed = np.meshgrid(lr_axis, seed_axis, indexing="ij")


# ---------------------------- postprocessing ----------------------------
# seeds are categorical, so use a discrete heatmap rather than a contour
fig, ax = plt.subplots(figsize=(2, 1), dpi=200)
ax.pcolormesh(np.log10(lr), seed, np.log10(val_grid), cmap="cividis")
best_lr_idx = np.argmin(val_grid, axis=0)  # lowest-cost lr per seed
ax.scatter(np.log10(lr_axis[best_lr_idx]), seed_axis, s=10, c="white")
ax.axis("off")
ax.set_rasterized(True)  # avoid contourline artifacts
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / "robustness_sine.png")
plt.close()
