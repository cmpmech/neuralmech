import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

from NN import DGCheb

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
EPOCHS = 1000
LR = 1e-2
K = 3

FREQUENCY = 7

cost_fun = nn.MSELoss()

CHANNELS = [8, 32, 32, 32, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(CHANNELS) - 2)]

# ------------------------------------ prepare data -----------------------------------
f = lambda x1, x2: torch.sin(FREQUENCY * 2 * torch.pi * x1 * x2)

data = torch.load(DATA_DIR / "ghana_mesh.pt", weights_only=False)
y = Data(
    x=f(data["pos"][:, 0], data["pos"][:, 1]).unsqueeze(-1),
    edge_index=data["edge_index"],
    pos=data["pos"],
).to(device)

x = Data(
    x=torch.randn((len(data["pos"]), CHANNELS[0])),
    edge_index=data["edge_index"],
    pos=data["pos"],
).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = DGCheb(CHANNELS, ACTIVATIONS, K=K).to(device)
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

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 10), dpi=100)
    ax.tripcolor(
        tri, y_pred.detach().squeeze().cpu(), shading="gouraud", cmap="Spectral"
    )
    ax.triplot(tri, color="k", linewidth=1, alpha=0.6)
    ax.set_aspect("equal")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:

    def save_tri(field, path):
        fig, ax = plt.subplots(figsize=(5, 10), dpi=200)
        ax.tripcolor(tri, field, shading="gouraud", cmap="Spectral")
        ax.triplot(tri, color="k", linewidth=1, alpha=0.6)
        ax.scatter(pos[:, 0], pos[:, 1], color="k", s=5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path, transparent=True)
        plt.close()

    save_tri(y.x.squeeze().cpu(), RGB_PDF_DIR / "GNN_cheb_target.pdf")
    save_tri(y_pred.detach().squeeze().cpu(), RGB_PDF_DIR / "GNN_cheb_prediction.pdf")
    for i in range(CHANNELS[0]):
        save_tri(
            x.x[:, i].detach().squeeze().cpu(), RGB_PDF_DIR / f"GNN_cheb_input_{i}.pdf"
        )

    fig, ax = plt.subplots(figsize=(5, 10), dpi=200)
    ax.triplot(tri, color="k", linewidth=1)
    ax.scatter(pos[:, 0], pos[:, 1], c=y.x.squeeze().cpu(), s=40, cmap="Spectral")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PDF_DIR / "GNN_cheb_target_points.pdf", transparent=True)
    plt.close()
