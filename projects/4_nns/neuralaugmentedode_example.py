import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from DL import init_weights
from NN import MLP, NODE

torch.manual_seed(0)
device = torch.device('cpu') # faster on a cpu

# -------------------------- training settings ---------------------------
resolution = 32

epochs = 100
lr = 2e-2

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
layers = [3, 32, 32, 2] # [3, 8, 8, 2] # CHANGED
activations = [torch.nn.GELU(approximate='tanh')] * (len(layers) - 2)

# ----------------------------- prepare data -----------------------------
x_ = np.sort(np.random.uniform(0, 1, resolution))
x_[0] = 0 # include 0 as initial condition
y_ = np.sin(2 * np.pi * x_)

x = torch.from_numpy(x_).float().to(device)
y = torch.from_numpy(y_).float().view(resolution, 1, 1).to(device) # (seq len, batch size, features)

# initial condition
y0 = y[0].repeat((1,2)) # CHANGED
# -------------------- instantiate model & optimizer ---------------------
rhs_model = MLP(layers, activations)
model = NODE(rhs_model)
model = model.to(device)
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
    y_pred = model(x, y0)[:, :, 0:1] # CHANGED
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({'train': f'{train_cost[epoch]:.2e}'})
toc = time.time()
print(f'elapsed time {toc - tic:.2f} s')

# ------------------------------ prediction ------------------------------
x = torch.linspace(0,1, 256, dtype=torch.float32).to(device)
y0 = torch.zeros((1 ,2), dtype=torch.float32).to(device)

y_pred = model(x, y0)[:,:,0:1] # CHANGED

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
plt.show()

# prediction
fig, ax = plt.subplots()
ax.plot(x_, y[:,0,0].cpu(), 'ko')
ax.plot(x.cpu(), y_pred[:,0,0].detach().cpu(), 'r--')
plt.show()