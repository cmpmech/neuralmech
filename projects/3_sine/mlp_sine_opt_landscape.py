import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import (
    Standardizer,
    filter_normalize_direction,
    flatten_params,
    get_params,
    init_weights,
    set_params,
    unflatten_params,
)
from NN import MLP

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")


CASE = 0
# CASE = 1

# -------------------------- training settings ---------------------------
EPOCHS = 500  # 500  # 4000
LR = 1e-2
BATCH_SIZE = 32
REGULARIZATION = 0.0
HIDDEN_LAYERS = 12  # 1 # 12

# define loss
cost_fun = nn.MSELoss(reduction="mean")


# ------------------------ interpolation settings ------------------------
GRID_STEPS = 200  # 200  # 40  # resolution of the landscape grid
ALPHA_RANGE = 2.0  # half-range along each direction

# ---------------------------- model settings ----------------------------
layers = [1] + [24] * HIDDEN_LAYERS + [1]
activations = [nn.Tanh()] * (len(layers) - 2)

# ----------------------------- prepare data -----------------------------
data = np.load(BASE_DIR / "../../data/sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.float32),
)

X_data = dataset.tensors[0]
Y_data = dataset.tensors[1]
standardizex = Standardizer(X_data, dim=0)
standardizey = Standardizer(Y_data, dim=0)

full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
num_samples = len(dataset)

# ----------------- instantiate model & prepare training -----------------
model = MLP(layers, activations).to(device)
init_weights(model, activations[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------- training -------------------------------
pbar = tqdm(range(EPOCHS), desc="training")
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = standardizex(x.to(device)), standardizey(y.to(device))
        optimizer.zero_grad()
        cost_fun(model(x), y).backward()
        optimizer.step()

# ------------------------ optimization landscape ------------------------
params_shape = get_params(model)
params_opt = flatten_params(params_shape)
dir1 = flatten_params(
    filter_normalize_direction(
        [torch.randn_like(p) for p in params_shape], params_shape
    )
)
dir2 = flatten_params(
    filter_normalize_direction(
        [torch.randn_like(p) for p in params_shape], params_shape
    )
)

a1 = np.linspace(-ALPHA_RANGE, ALPHA_RANGE, GRID_STEPS)
a2 = np.linspace(-ALPHA_RANGE, ALPHA_RANGE, GRID_STEPS)
a1, a2 = np.meshgrid(a1, a2, indexing="ij")
costs = np.zeros((GRID_STEPS, GRID_STEPS))

model.eval()
for i in tqdm(range(GRID_STEPS), desc="sampling"):
    for j in range(GRID_STEPS):
        cur = params_opt + a1[i, j] * dir1 + a2[i, j] * dir2
        set_params(model, unflatten_params(cur, params_shape))
        with torch.no_grad():
            x, y = next(iter(full_loader))
            idx = num_samples // 2
            x = x[idx * CASE : idx * (CASE + 1)]
            y = y[idx * CASE : idx * (CASE + 1)]
            x, y = standardizex(x), standardizey(y)
            costs[i, j] = cost_fun(model(x), y).item()

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=200)  # TODO
log_costs = np.log10(costs + 1e-10)
ax.contourf(a1, a2, log_costs, levels=60, cmap="cividis")
ax.plot(0, 0, "ws", ms=4)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)  # avoid contourline artifacts
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if args.book:
# ------------------------- book postprocessing --------------------------
    fig.savefig(RESULTS_DIR / f"NN_landscape_{HIDDEN_LAYERS}_{CASE}.pdf")
else:
    plt.show()
plt.close(fig)
