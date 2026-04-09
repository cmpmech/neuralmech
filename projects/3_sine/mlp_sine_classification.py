from pathlib import Path

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from postprocessing import save_csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from DL import init_weights, Standardizer
from NN import MLP
import time

BASE_DIR = Path(__file__).parent
torch.manual_seed(0)
device = torch.device('cpu')  # faster on cpu, because matrices are small

# -------------------------- training settings ---------------------------
epochs = 200
lr = 1e-2
regularization = 1e-2
batch_size = 32

# define loss
cost_fun = nn.CrossEntropyLoss(reduction='mean')

# ---------------------------- model settings ----------------------------
classes = 3 # has to fit to the data generation
layers = [1, 24, 24, 24, classes] # three hidden layers is sufficient (five to show overfitting)
activations = [torch.nn.GELU(approximate='tanh')] * (len(layers) - 2)

# ----------------------------- prepare data -----------------------------
data = np.load(BASE_DIR / '../../data/discrete_sine.npz')
dataset = TensorDataset(torch.from_numpy(data['X']).to(torch.float32),
                        torch.from_numpy(data['Y']).to(torch.long))
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

test_data = np.load(BASE_DIR / '../../data/discrete_sine_test.npz')
x_test = torch.from_numpy(test_data['X']).to(torch.float32).to(device)
y_test = torch.from_numpy(test_data['Y']).to(torch.long).to(device)

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=0)

# ----------------- instantiate model & prepare training -----------------
model = MLP(layers, activations)
model.to(device)
init_weights(model, activations[0])
optimizer = torch.optim.AdamW(model.parameters(), lr,
                              weight_decay=regularization)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

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
        x = standardizex(x)
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
            x = standardizex(x)
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
print(f'Elapsed time {toc - tic:.2f} s')

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
ax.plot(val_cost, 'r')
plt.show()

model.eval()
with torch.no_grad():
    y_pred_test = torch.argmax(model(standardizex(x_test)), dim=-1)

Y_train = train_data.dataset.tensors[1][train_data.indices]
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]
fig, ax = plt.subplots()
ax.plot(x_test.cpu(), y_test.cpu(), 'k')
ax.plot(X_train, Y_train, 'ko')
ax.plot(X_val, Y_val, 'ro')
ax.plot(x_test.cpu(), y_pred_test.cpu(), 'b.')
plt.show()

# ------------------------- book post-processing -------------------------
save_csv(BASE_DIR / '../../results/mlp_discrete_sine_test.csv', x=x_test[:,0],
         y=y_test, ypred=y_pred_test)
save_csv(BASE_DIR / '../../results/mlp_discrete_sine_train.csv', x=X_train[:,0], y=Y_train)
save_csv(BASE_DIR / '../../results/mlp_discrete_sine_val.csv', x=X_val[:,0], y=Y_val)