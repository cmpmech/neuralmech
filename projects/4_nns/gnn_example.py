from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data
from NN import DGSAGE
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------- training settings ---------------------------
epochs = 1000
lr = 1e-2

# define cost
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
channels = [8, 32, 32, 32, 1]
activations = [torch.nn.GELU(approximate='tanh')] * (len(channels) - 2)

# ----------------------------- prepare data -----------------------------
f = lambda x1, x2 : torch.sin(14 * torch.pi * x1 * x2)

data = torch.load(BASE_DIR / '../../data/ghana_mesh.pt')
y = Data(x=f(data['pos'][:,0], data['pos'][:,1]).unsqueeze(-1),
         edge_index=data['edge_index'],
         pos=data['pos']).to(device)

# -------------------- instantiate model & optimizer ---------------------
model = DGSAGE(channels, activations).to(device)
x = Data(x=torch.randn((len(data['pos']), channels[0])),
         edge_index=data['edge_index'],
         pos=data['pos']).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr)

# ------------------------------- training -------------------------------
train_cost = [0] * epochs
tic = time.time()
print_every = 10
pbar = tqdm(range(epochs))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x)
    cost = cost_fun(y_pred, y.x)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({'train': f'{train_cost[epoch]:.2e}'})
toc = time.time()
print(f'elapsed time {toc - tic:.2f} s')

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
plt.show()

pos = y.pos.cpu().numpy()
edge_index = y.edge_index.cpu().numpy().T
triangles = data['triangles'].cpu().numpy()

tri = mtri.Triangulation(pos[:,0], pos[:,1], triangles)

# ground truth
fig, ax = plt.subplots(figsize=(5,10), dpi=200)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.tripcolor(tri, y.x.squeeze().cpu(), shading='gouraud', cmap='Spectral')
ax.triplot(tri, color='k', linewidth=1, alpha=0.6)
ax.scatter(pos[:,0], pos[:,1], color='k', s=5)
plt.gca().set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(BASE_DIR / f'../../results/GNN_target.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()

# prediction
fig, ax = plt.subplots(figsize=(5,10), dpi=200)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.tripcolor(tri, y_pred.detach().squeeze().cpu(), shading='gouraud', cmap='Spectral')
ax.triplot(tri, color='k', linewidth=1, alpha=0.6)
ax.scatter(pos[:,0], pos[:,1], color='k', s=5)
plt.gca().set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(BASE_DIR / f'../../results/GNN_prediction.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()

for i in range(channels[0]):
    fig, ax = plt.subplots(figsize=(5,10), dpi=200)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.tripcolor(tri, x.x[:,i].detach().squeeze().cpu(), shading='gouraud', cmap='Spectral')
    ax.triplot(tri, color='k', linewidth=1, alpha=0.6)
    ax.scatter(pos[:,0], pos[:,1], color='k', s=5)
    plt.gca().set_aspect('equal')
    ax.axis('off')
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(BASE_DIR / f'../../results/GNN_input_{i}.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

fig, ax = plt.subplots(figsize=(5,10), dpi=200)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.triplot(tri, color='k', linewidth=1, alpha=1)
ax.scatter(pos[:,0], pos[:,1], c=y.x.squeeze().cpu(), s=40, cmap='Spectral')
plt.gca().set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(BASE_DIR / f'../../results/GNN_target_points.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()