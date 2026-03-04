import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from NN import DCN
from DL import init_weights

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------- training settings ---------------------------
resolution = 256

epochs = 2000
lr = 2e-2

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
channels = [16, 16, 16, 16, 1]
activations = [torch.nn.GELU(approximate='tanh')] * (len(channels) - 2)
normalizations = [nn.GroupNorm(1, channels[i + 1]) for i in range(len(channels) - 2)] # equivalent LayerNorm without specifying image size
resamplings = [nn.Upsample(scale_factor=2, mode='bilinear')] * (len(channels) - 2)
kernel_size, stride, padding = 3, 1, 1

# ----------------------------- prepare data -----------------------------
x1 = np.linspace(-1, 1, resolution)
x2 = np.linspace(-1, 1, resolution)
x1, x2 = np.meshgrid(x1, x2, indexing='ij')

y = np.sin(2 * np.pi * x1) * np.sin(12 * np.pi * x1 * x2)

y = torch.from_numpy(y).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)

# -------------------- instantiate model & optimizer ---------------------
model = DCN(channels, activations,
            kernel_size, stride, padding,
            normalizations=normalizations, resamplings=resamplings,).to(device)
input_size = resolution // 2 ** (len(channels) - 2)
x = torch.randn((1, channels[0], input_size, input_size),
                dtype=torch.float32).to(device)

init_weights(model, activations[0])
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
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({'train': f'{train_cost[epoch]:.2e}'})
toc = time.time()
print(f'elapsed time {toc - tic:.2f} s')

# --------------------------- post-processing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
plt.show()

# prediction
fig, ax = plt.subplots(figsize=(resolution/100,resolution/100), dpi=100)
cb = ax.pcolormesh(x1, x2, y_pred[0,0].detach().cpu(), vmin=-1, vmax=1,
                   cmap='Spectral')
# fig.colorbar(cb)
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/CNN_prediction.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# ground truth
fig, ax = plt.subplots(figsize=(resolution/100,resolution/100), dpi=100)
cb = ax.pcolormesh(x1, x2, y[0,0].detach().cpu(), vmin=-1, vmax=1,
                   cmap='Spectral')
# fig.colorbar(cb)
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/CNN_target.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# input
for i in range(channels[0]):
    fig, ax = plt.subplots(figsize=(input_size/10, input_size/10), dpi=100)
    ax.imshow(x[0,i].detach().cpu().T, origin='lower', vmin=-1, vmax=1,
              cmap='Spectral')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(f'../../results/CNN_input_{i}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()