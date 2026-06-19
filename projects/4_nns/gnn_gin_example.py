import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

from NN import DGIN

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 1000
LR = 1e-2

# define loss
cost_fun = nn.MSELoss()

# model settings
# one message-passing block, each parametrized by its own MLP
MLP_LAYERS = [[8, 24, 24, 24, 24, 1]]
MLP_ACTIVATIONS = [[nn.GELU(approximate="tanh") for _ in range(len(MLP_LAYERS[0]) - 2)]]

# ------------------------------------ prepare data -----------------------------------
f = lambda x1, x2: torch.sin(14 * torch.pi * x1 * x2)

data = torch.load(DATA_DIR / "ghana_mesh.pt", weights_only=False)
y = Data(
    x=f(data["pos"][:, 0], data["pos"][:, 1]).unsqueeze(-1),
    edge_index=data["edge_index"],
    pos=data["pos"],
).to(device)

# random noise on the nodes mapped to the target
x = Data(
    x=torch.randn((len(data["pos"]), MLP_LAYERS[0][0])),
    edge_index=data["edge_index"],
    pos=data["pos"],
).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = DGIN(MLP_LAYERS, MLP_ACTIVATIONS, train_eps=True).to(device)
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
    cost = cost_fun(y_pred, y.x)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
pos = y.pos.cpu().numpy()
triangles = data["triangles"].cpu().numpy()
tri = mtri.Triangulation(pos[:, 0], pos[:, 1], triangles)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(train_cost, "k")
plt.show()

fig, ax = plt.subplots(figsize=(5, 10), dpi=100)
ax.tripcolor(tri, y_pred.detach().squeeze().cpu(), shading="gouraud", cmap="Spectral")
ax.triplot(tri, color="k", linewidth=1, alpha=0.6)
ax.set_aspect("equal")
plt.show()
