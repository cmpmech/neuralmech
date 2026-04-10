import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from NN import DRNN
from DL import init_weights
from postprocessing import save_csv

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------- training settings ---------------------------
resolution = 256

epochs = 3000
lr = 1e-3

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
cell = nn.RNN
# cell = nn.LSTM
# cell = nn.GRU
layers = [1, 16, 16, 1]

if cell == nn.GRU:
    lr = 2e-3

# ----------------------------- prepare data -----------------------------
x_ = np.linspace(0, 1, resolution)
y_ = 0.5*(np.sin(10 * np.pi * x_) + np.sin(23.7 * np.pi * x_)) # cannot be integer

x = torch.ones(1, resolution, 1, dtype=torch.float32).to(device) * 0.5
x[:,0,:] = 1.
y = torch.from_numpy(y_).float().view(1, resolution, 1).to(device)

# -------------------- instantiate model & optimizer ---------------------
model = DRNN(layers, cell=cell)
model.to(device)
init_weights(model, nn.Tanh())
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
cell2string = {nn.RNN : 'rnn', nn.LSTM : 'lstm', nn.GRU : 'gru'}

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
plt.show()

# prediction
fig, ax = plt.subplots(figsize=(5,4), dpi=100)
ax.plot(x_, y_pred[0,:,0].detach().cpu(), 'k', linewidth=1.5)
ax.plot(x_, y[0,:,0].detach().cpu(), 'r--', linewidth=1.5)
fig.tight_layout(pad=0)
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/rnn_sine_{cell2string[cell]}.csv',
         i=np.arange(resolution) + 1, x=x.squeeze().detach().cpu(),
         z=x_, y=y.squeeze().detach().cpu(),
         y_pred=y_pred.squeeze().detach().cpu())