import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from NN import MLP
from DL import init_weights

torch.manual_seed(0)
device = torch.device('cpu')

# -------------------------- training settings ---------------------------
resolution = 256

epochs = 200
lr = 2e-2

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
layers = [8, 32, 32, 32, 1]
activations = [torch.nn.GELU(approximate='tanh')] * (len(layers) - 2)

# ----------------------------- prepare data -----------------------------
x_ = np.linspace(-1, 1, resolution)
y = np.sin(8 * np.pi * x_) * np.sin(6 * np.pi * x_)

y = torch.from_numpy(y).to(torch.float32).unsqueeze(1).to(device).to(device)

# -------------------- instantiate model & optimizer ---------------------
model = MLP(layers, activations)
model.to(device)
x = torch.randn((resolution, layers[0]), dtype=torch.float32).to(device)
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

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
plt.show()

# prediction
fig, ax = plt.subplots(figsize=(5,4), dpi=100)
# fig.patch.set_alpha(0)
# ax.patch.set_alpha(0)
ax.plot(x_, y_pred.detach().cpu(), 'k', linewidth=1.5)
ax.axis('off')
fig.tight_layout(pad=0)
plt.savefig(f'../../results/MLP_prediction.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# ground truth
fig, ax = plt.subplots(figsize=(5,4), dpi=100)
# fig.patch.set_alpha(0)
# ax.patch.set_alpha(0)
ax.plot(x_, y.detach().cpu(), 'k', linewidth=1.5)
ax.axis('off')
fig.tight_layout(pad=0)
plt.savefig(f'../../results/MLP_target.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

#input
fig, ax = plt.subplots(figsize=(5,6), dpi=100)
# fig.patch.set_alpha(0)
# ax.patch.set_alpha(0)
for i in range(layers[0]):
    ax.plot(x_+0.1*i, x[:,i].detach().cpu()-3*i, 'k', linewidth=1.5)
ax.axis('off')
fig.tight_layout(pad=0)
plt.savefig(f'../../results/MLP_input.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
