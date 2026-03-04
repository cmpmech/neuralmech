import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from postprocessing import save_csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from DL import init_weights, Standardizer
from NN import MLP
import time

torch.manual_seed(0)
device = torch.device('cpu')  # faster on cpu, because matrices are small

# -------------------------- training settings ---------------------------
epochs = 4000 # 400
lr = 1e-2
regularization = 0 #1e0
batch_size = 32

# define loss
cost_fun = nn.MSELoss(reduction='mean')

# ---------------------------- model settings ----------------------------
layers = [1, 24, 24, 24, 1] # three hidden layers is sufficient (five to show overfitting)
activations = [nn.GELU(approximate='tanh')] * (len(layers) - 2)

# ----------------------------- prepare data -----------------------------
data = np.load('../../data/sine.npz')
dataset = TensorDataset(torch.from_numpy(data['X']).to(torch.float32),
                        torch.from_numpy(data['Y']).to(torch.float32))
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

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
ax.set_yscale('log')
ax.plot(train_cost, 'k')
ax.plot(val_cost, 'r')
plt.show()

x_test = torch.linspace(-1.3, 1.3, 100).unsqueeze(1)
y_test = torch.sin(2 * torch.pi * x_test)

model.eval()
with torch.no_grad():
    y_pred_test = model(standardizex(x_test).to(device))
    y_pred_test = standardizey.inverse(y_pred_test).cpu().numpy()

fig, ax = plt.subplots()
ax.plot(x_test, y_test, 'k')
ax.plot(X_train, Y_train, 'ko')
X_val = train_data.dataset.tensors[0][val_data.indices]
Y_val = train_data.dataset.tensors[1][val_data.indices]
ax.plot(X_val, Y_val, 'ro')
ax.plot(x_test, y_pred_test, 'r--')
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/mlp_sine_test_{epochs}.csv',
         x=x_test.squeeze(), y=y_test[:,0], ypred=y_pred_test[:,0])
save_csv(f'../../results/mlp_sine_train.csv',
         x=X_train[:, 0], y=Y_train[:, 0])
save_csv(f'../../results/mlp_sine_val.csv',
         x=X_val[:, 0], y=Y_val[:, 0])
if epochs == 4000:
    save_csv(f'../../results/mlp_sine_cost_history.csv',
             train=np.array(train_cost) / train_cost[0],
             val=np.array(val_cost) / val_cost[0])