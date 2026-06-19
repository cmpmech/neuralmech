import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import PointNetConv, fps, global_max_pool, knn_interpolate, radius
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


# ------------------------------------ pointnet++ -------------------------------------
class SAModule(nn.Module):
    def __init__(self, ratio, radius, local_mlp):
        super().__init__()
        self.ratio = ratio
        self.radius = radius
        self.conv = PointNetConv(local_mlp, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)  # downsample centroids
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)  # neighbours -> centroids
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x, pos[idx], batch[idx]


class GlobalSAModule(nn.Module):
    def __init__(self, global_mlp):
        super().__init__()
        self.global_mlp = global_mlp

    def forward(self, x, pos, batch):
        x = self.global_mlp(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.shape[0], 3))
        batch = torch.arange(x.shape[0], device=batch.device)
        return x, pos, batch


class FPModule(nn.Module):
    def __init__(self, k, fp_mlp):
        super().__init__()
        self.k = k
        self.fp_mlp = fp_mlp

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return self.fp_mlp(x), pos_skip, batch_skip


class PointNetPP(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.sa1 = SAModule(0.5, 0.2, MLP([3, 32, 32], [activation, activation]))
        self.sa2 = SAModule(0.25, 0.4, MLP([32 + 3, 64, 64], [activation, activation]))
        self.sa3 = GlobalSAModule(MLP([64 + 3, 128, 128], [activation, activation]))

        self.fp3 = FPModule(3, MLP([128 + 64, 64], [activation]))
        self.fp2 = FPModule(3, MLP([64 + 32, 32], [activation]))
        self.fp1 = FPModule(3, MLP([32, 32], [activation]))

        self.head = MLP([32, 64, 1], [activation, None])

    def forward(self, pos):
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        sa0 = (None, pos, batch)
        sa1 = self.sa1(*sa0)
        sa2 = self.sa2(*sa1)
        sa3 = self.sa3(*sa2)

        x, _, _ = self.fp3(*sa3, *sa2)
        x, _, _ = self.fp2(x, sa2[1], sa2[2], *sa1[:1], sa1[1], sa1[2])
        x, _, _ = self.fp1(x, sa1[1], sa1[2], None, sa0[1], sa0[2])
        return self.head(x)


# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000
LR = 4e-3

# define loss
cost_fun = nn.MSELoss()

# model settings
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
model = PointNetPP(ACTIVATION).to(device)
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
    "bunny pointnet++ regression",
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
vtk_out = RESULTS_DIR / "pointnetpp_bunny.vtk"
vtk_out.write_text("\n".join(lines) + "\n")

print(f"\tfinal cost {train_cost[-1]:.2e}\n\tsaved {vtk_out}")
