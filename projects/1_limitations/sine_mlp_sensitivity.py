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

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 800
REGULARIZATION = 0
BATCH_SIZE = 32

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
# architecture fixed: only the init seed and learning rate vary
LAYERS = [1, 12, 12, 12, 1]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]

# sweep
SEED_COUNT = 100
LR_RESOLUTION = 200
LR_RANGE = (1e-4, 1e0)

seed_axis = np.arange(SEED_COUNT)
lr_axis = np.logspace(np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1]), LR_RESOLUTION)

# ------------------------------------ prepare data -----------------------------------
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


# ---------------------------------- single training ----------------------------------
def train_run(seed, lr):
    torch.manual_seed(seed)  # only the init + shuffling stochasticity changes
    model = MLP(LAYERS, ACTIVATIONS).to(device)
    init_weights(model, ACTIVATIONS[0])
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


# --------------------------------------- sweep ---------------------------------------
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

# ----------------------------------- postprocessing ----------------------------------
lr, seed = np.meshgrid(lr_axis, seed_axis, indexing="ij")

# seeds are categorical, so use a discrete heatmap rather than a contour
fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
ax.pcolormesh(np.log10(lr), seed, np.log10(val_grid), cmap="cividis")
best_lr_ids = np.argmin(val_grid, axis=0)  # lowest-cost lr per seed
ax.scatter(np.log10(lr_axis[best_lr_ids]), seed_axis, s=1, c="white")
ax.axis("off")
ax.set_rasterized(True)  # avoid contourline artifacts
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig.savefig(RESULTS_DIR / "robustness_sine.png")
plt.close()
