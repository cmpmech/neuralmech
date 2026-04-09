from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data
from NN import DGIN
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from tqdm import tqdm

# ---------------------------------- NN ----------------------------------

BASE_DIR = Path(__file__).parent
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------- training settings ---------------------------
epochs = 1000
lr = 1e-2

# define cost
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
mlps = 1
layers = [1, 24, 24, 24, 24, 1]
mlp_layers = [layers * mlps]
mlp_layers[0][0] = 8 # first layer
mlp_activations = [[nn.GELU(approximate='tanh')] * (len(layers) - 2)]

# ----------------------------- prepare data -----------------------------
f = lambda x1, x2 : torch.sin(14 * torch.pi * x1 * x2)

data = torch.load(BASE_DIR / '../../data/ghana_mesh.pt')
y = Data(x=f(data['pos'][:,0], data['pos'][:,1]).unsqueeze(-1),
         edge_index=data['edge_index'],
         pos=data['pos']).to(device)

# -------------------- instantiate model & optimizer ---------------------
model = DGIN(mlp_layers, mlp_activations, train_eps=True).to(device)

x = Data(x=torch.randn((len(data['pos']), mlp_layers[0][0])),
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
plt.show()