import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results/3D").resolve()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()


# ------------------------------------- pointnet --------------------------------------
class PointNet(nn.Module):
    def __init__(self, point_layers, global_layers, head_layers, activation):
        super().__init__()
        self.point_mlp = MLP(point_layers, [activation] * (len(point_layers) - 1))
        self.global_mlp = MLP(global_layers, [activation] * (len(global_layers) - 1))
        self.head_mlp = MLP(head_layers, [activation] * (len(head_layers) - 2))

    def forward(self, x):
        l = self.point_mlp(x)
        g = l.max(dim=0, keepdim=True).values  # global max pool
        g = self.global_mlp(g)
        g = g.expand(x.shape[0], -1)  # prepare concatenation
        return self.head_mlp(torch.cat([l, g], dim=1))


# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000
LR = 4e-3

# define loss
cost_fun = nn.MSELoss()

# model settings
POINT_LAYERS = [3, 32, 32]
GLOBAL_LAYERS = [32, 32]
HEAD_LAYERS = [POINT_LAYERS[-1] + GLOBAL_LAYERS[-1], 64, 1]
ACTIVATION = nn.GELU(approximate="tanh")

# ------------------------------------ prepare data -----------------------------------
FREQ = 2.0
f = lambda p: (
    torch.sin(FREQ * torch.pi * p[:, 0])  # * torch.sin(FREQ * torch.pi * p[:, 1])
    # * torch.sin(FREQ * torch.pi * p[:, 2])
)

points = np.load(DATA_DIR / "bunny_pointcloud.npz")["points"]
points = torch.from_numpy(points).to(torch.float32)

center = points.mean(dim=0)
scale = (points - center).abs().max()
xn = ((points - center) / scale).to(device)  # normalized coords in ~[-1, 1]

x = xn
y = f(xn).unsqueeze(-1)

# --------------------------- instantiate model & optimizer ---------------------------
model = PointNet(POINT_LAYERS, GLOBAL_LAYERS, HEAD_LAYERS, ACTIVATION).to(device)
init_weights(model, ACTIVATION)
optimizer = torch.optim.AdamW(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x)
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

truth = y.squeeze().cpu().numpy()
pred = y_pred.detach().squeeze().cpu().numpy()
error = pred - truth
coords = points.numpy()
xn = xn.cpu().numpy()

m = len(coords)
lines = [
    "# vtk DataFile Version 3.0",
    "bunny pointnet regression",
    "ASCII",
    "DATASET POLYDATA",
    f"POINTS {m} float",
]
lines.extend(f"{px} {py} {pz}" for px, py, pz in coords)
lines.append(f"VERTICES {m} {2 * m}")
lines.extend(f"1 {i}" for i in range(m))
lines.append(f"POINT_DATA {m}")
for name, field in (("ground_truth", truth), ("prediction", pred), ("error", error)):
    lines.append(f"SCALARS {name} float 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(f"{v}" for v in field)
lines.append("VECTORS input float")
lines.extend(f"{px} {py} {pz}" for px, py, pz in xn)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
vtk_out = RESULTS_DIR / "pointnet_bunny.vtk"
vtk_out.write_text("\n".join(lines) + "\n")

print(f"\tfinal cost {train_cost[-1]:.2e}\n\tsaved {vtk_out}")
