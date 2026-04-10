import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import time
from tqdm import tqdm
from NN import KAN

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------- training settings ---------------------------
resolution = 256

epochs = 100
lr = 4e-2

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
layers = [8, 8, 8, 1]
spline_order, grid_size = 3, 5 # default is 3, 5
base_activation = nn.SiLU

# ----------------------------- prepare data -----------------------------
x_ = np.linspace(-1, 1, resolution)
y = np.sin(8 * np.pi * x_) * np.sin(6 * np.pi * x_)

y = torch.from_numpy(y).to(torch.float32).unsqueeze(1).to(device).to(device)

# -------------------- instantiate model & optimizer ---------------------
model = KAN(layers, spline_order=spline_order, grid_size=grid_size,
            base_activation=base_activation)
model.to(device)
x = torch.randn((resolution, layers[0]), dtype=torch.float32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)

print(model)

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
fig, ax = plt.subplots(dpi=200)
ax.plot(x_, y.cpu(), 'k', linewidth=1.5)
ax.plot(x_, y_pred.detach().cpu(), 'r--', linewidth=1.5)
plt.show()

# ------------------------------- testing --------------------------------
def count_kan_parameters(model):
    base_params = 0
    spline_params = 0

    for name, param in model.named_parameters():
        if 'base_weight' in name:
            base_params += param.numel()
        elif 'spline_weight' in name or 'spline_scaler' in name:
            spline_params += param.numel()

    print(f"{'Component':<20} | {'Count':<10}")
    print("-" * 35)
    print(f"{'Linear (Base)':<20} | {base_params:<10}")
    print(f"{'Spline (Activation)':<20} | {spline_params:<10}")
    print("-" * 35)
    print(f"{'Total Trainable':<20} | {base_params + spline_params:<10}")

count_kan_parameters(model)








