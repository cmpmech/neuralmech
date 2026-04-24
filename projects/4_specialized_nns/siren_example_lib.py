from pathlib import Path

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from postprocessing import save_csv
import matplotlib.pyplot as plt
from DL import Standardizer
from siren_pytorch import SirenNet
import time
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cpu')

# -------------------------- training settings ---------------------------
epochs = 1000
lr = 1e-4
batch_size = 32

cost_fun = nn.MSELoss(reduction='mean')

# ---------------------------- model settings ----------------------------
dim_hidden = 32
num_layers = 2   # number of hidden layers; output layer is added automatically
w0_initial = 5.0
w0 = 1.0

# ----------------------------- prepare data -----------------------------
data = np.load(BASE_DIR / '../../data/sine.npz')
dataset = TensorDataset(torch.from_numpy(data['X']).to(torch.float32),
                        torch.from_numpy(data['Y']).to(torch.float32))
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

# -------------------- instantiate model & optimizer ---------------------
model = SirenNet(dim_in=1, dim_hidden=dim_hidden, dim_out=1,
                 num_layers=num_layers, w0_initial=w0_initial, w0=w0)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ------------------------------- training -------------------------------
train_cost = [0.0] * epochs
tic = time.time()
pbar = tqdm(range(epochs))
model.train()
for epoch in pbar:
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x, y = standardizex(x), standardizey(y)
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    if epoch % 50 == 0:
        pbar.set_postfix({'train': f'{train_cost[epoch]:.2e}'})
toc = time.time()
print(f'elapsed time: {toc - tic:.2f} s')

# ---------------------------- postprocessing ----------------------------
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]

model.eval()
with torch.no_grad():
    y_val_pred = standardizey.inverse(model(standardizex(X_val)))
    cost = cost_fun(y_val_pred, Y_val)
print(f'validation cost: {cost:.2e}')

f = lambda x: torch.sin(2 * torch.pi * x)
x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1)
y_test = f(x_test)

with torch.no_grad():
    y_pred = standardizey.inverse(model(standardizex(x_test)))

fig, ax = plt.subplots()
ax.plot(x_test, y_test, 'k')
ax.plot(x_test, y_pred.detach().cpu(), 'r--')
ax.plot(X_train, Y_train, 'bo')
ax.set_ylim(-2, 2)
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(BASE_DIR / '../../results/siren_lib_sine_test.csv',
         x=x_test[:, 0], y=y_test[:, 0], ypred=y_pred[:, 0])
save_csv(BASE_DIR / '../../results/siren_lib_sine_train.csv',
         x=X_train[:, 0], y=Y_train[:, 0])
