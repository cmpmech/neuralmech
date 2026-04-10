import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from NN import MLP
from DL import Standardizer, init_weights
import time
from tqdm import tqdm

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cpu')

# -------------------------- training settings ---------------------------
epochs = 1000
lr = 1e-2
regularization = 0
batch_size = 32

# define loss
cost_fun = nn.MSELoss(reduction='mean')

# ---------------------------- model settings ----------------------------
use_symmetry = True
use_structure = True

layers = [1, 24, 24, 1]
activations = [torch.nn.GELU(approximate='tanh')] * (len(layers) - 2)

# ----------------------------- prepare data -----------------------------
num_samples = 96
x1 = torch.rand(num_samples, 1) * 2 - 1
x2 = torch.rand(num_samples, 1) * 2 - 1
y = torch.sin(2 * torch.pi * x1) * torch.sin(2 * torch.pi * x2)
# y = torch.sqrt(torch.sin(2 * torch.pi * x1) * torch.sin(2 * torch.pi * x2) + 1.)

dataset = TensorDataset(torch.cat([x1, x2], 1), y)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=(0,1)) # would break symmetry
standardizey = Standardizer(Y_train, dim=0)

# -------------------- instantiate model & optimizer ---------------------
class SymmetricMLP(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder # shared among all inputs

    def forward(self, x):
        x1, x2 = x[:,0:1], x[:,1:2]
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return h1 * h2

class UnSymmetricMLP(nn.Module):
    def __init__(self, encoder1, encoder2):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def forward(self, x):
        x1, x2 = x[:,0:1], x[:,1:2]
        h1 = self.encoder1(x1)
        h2 = self.encoder2(x2)
        return h1 * h2

if not use_structure:
    model = MLP([2] + layers[1:], activations)
    init_weights(model, activations[0])
elif not use_symmetry:
    encoder1 = MLP(layers, activations)
    encoder2 = MLP(layers, activations)
    init_weights(encoder1, activations[0])
    init_weights(encoder2, activations[0])
    model = UnSymmetricMLP(encoder1, encoder2)
else:
    encoder = MLP(layers, activations)
    init_weights(encoder, activations[0])
    model = SymmetricMLP(encoder)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr,
                              weight_decay=regularization)

# ------------------------------- training -------------------------------
train_cost = [0] * epochs
val_cost = [0] * epochs
tic = time.time()
print_every = 10
pbar = tqdm(range(epochs))
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x, y = standardizex(x), standardizey(y)
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader) # avg per batch

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x, y = standardizex(x), standardizey(y)
            y_pred = model(x)
            cost = cost_fun(y_pred, y)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader) # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix({
            'train': f'{train_cost[epoch]:.2e}',
            'val': f'{val_cost[epoch]:.2e}'
        })
toc = time.time()
print(f'elapsed time {toc - tic:.2f} s')

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, 'r')
ax.plot(val_cost, 'b')
ax.set_yscale('log')
plt.show()

resolution = 300
x1 = torch.linspace(-1, 1, resolution)
x2 = torch.linspace(-1, 1, resolution)
x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
with torch.no_grad():
    model_input = torch.cat([x1.flatten().unsqueeze(1),
                                    x2.flatten().unsqueeze(1)], 1)
    y = standardizey.inverse(model(standardizex(model_input)))

fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
ax.pcolormesh(x1, x2, y.reshape(resolution,
                                      resolution),
                                      cmap='Spectral', vmin=-1, vmax=1)
ax.plot(X_train[:,0], X_train[:,1], 'ko', markersize=4, alpha=0.4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/paramsharing_{use_symmetry}_{use_structure}.pdf',
                  bbox_inches='tight', pad_inches=0)
plt.show()

fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
ax.pcolormesh(x1, x2, y.reshape(resolution,
                                      resolution).T,
                                      cmap='Spectral', vmin=-1, vmax=1)
ax.plot(X_train[:,0], X_train[:,1], 'ko', markersize=4, alpha=0.4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/paramsharingT_{use_symmetry}_{use_structure}.pdf',
                  bbox_inches='tight', pad_inches=0)
plt.show()