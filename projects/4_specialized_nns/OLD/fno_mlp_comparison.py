import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralop.models import FNO
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- training settings ---------------------------
resolution_train = 256
resolution_test = 1024
frequency = 10.0
epochs = 200
lr = 1e-3
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
# FNO
k_modes = 16
hidden_channels = 32
n_layers = 2

# MLP
mlp_layers = [1, 64, 64, 64, 1]
mlp_activations = [nn.GELU(approximate="tanh")] * (len(mlp_layers) - 2)


# ----------------------------- prepare data -----------------------------
def make_data(resolution, frequency, device):
    z = np.linspace(-1, 1, resolution)
    x = np.sin(frequency * z)
    y = np.sin(frequency * (z + 0.5 * np.pi))
    z_t = (
        torch.from_numpy(z).to(torch.float32).unsqueeze(1).to(device)
    )  # (n, 1) for MLP
    x_fno = (
        torch.from_numpy(x).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )  # (1,1,n) for FNO
    y_fno = torch.from_numpy(y).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    y_mlp = (
        torch.from_numpy(y).to(torch.float32).unsqueeze(1).to(device)
    )  # (n, 1) for MLP
    return z, z_t, x_fno, y_fno, y_mlp


z_train, z_t_train, x_fno_train, y_fno_train, y_mlp_train = make_data(
    resolution_train, frequency, device
)
z_test, z_t_test, x_fno_test, y_fno_test, y_mlp_test = make_data(
    resolution_test, frequency, device
)

# -------------------- instantiate models & optimizers -------------------
fno = FNO(
    n_modes=(k_modes,),
    in_channels=1,
    out_channels=1,
    hidden_channels=hidden_channels,
    n_layers=n_layers,
).to(device)

mlp = MLP(mlp_layers, mlp_activations).to(device)
init_weights(mlp, mlp_activations[0])

optimizer_fno = torch.optim.Adam(fno.parameters(), lr=lr)
optimizer_mlp = torch.optim.AdamW(mlp.parameters(), lr=lr)


# ------------------------------- training -------------------------------
def train(model, optimizer, x, y, epochs, label):
    costs = []
    pbar = tqdm(range(epochs), desc=label)
    model.train()
    tic = time.time()
    for epoch in pbar:
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        costs.append(cost.item())
        if epoch % 50 == 0:
            pbar.set_postfix({"loss": f"{cost.item():.2e}"})
    print(f"{label} elapsed: {time.time() - tic:.2f} s")
    return costs


costs_fno = train(fno, optimizer_fno, x_fno_train, y_fno_train, epochs, "FNO")
costs_mlp = train(mlp, optimizer_mlp, z_t_train, y_mlp_train, epochs, "MLP")

# ---------------------------- postprocessing ----------------------------
# loss curves
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(costs_fno, "b", label="FNO")
ax.plot(costs_mlp, "r", label="MLP")
ax.legend()
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
plt.show()

# predictions at training resolution
fno.eval()
mlp.eval()
with torch.no_grad():
    y_pred_fno_train = fno(x_fno_train).squeeze().cpu()
    y_pred_mlp_train = mlp(z_t_train).squeeze().cpu()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
for ax, pred, title in zip(axes, [y_pred_fno_train, y_pred_mlp_train], ["FNO", "MLP"]):
    ax.plot(z_train, y_fno_train.squeeze().cpu(), "k", lw=1.5, label="target")
    ax.plot(z_train, pred, "r--", lw=1.5, label="prediction")
    ax.set_title(f"{title} — train res {resolution_train}")
    ax.legend()
plt.tight_layout()
plt.show()

# resolution generalization
with torch.no_grad():
    y_pred_fno_test = fno(x_fno_test).squeeze().cpu()
    y_pred_mlp_test = mlp(z_t_test).squeeze().cpu()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
for ax, pred, title in zip(axes, [y_pred_fno_test, y_pred_mlp_test], ["FNO", "MLP"]):
    ax.plot(z_test, y_fno_test.squeeze().cpu(), "k", lw=1.5, label="target")
    ax.plot(z_test, pred, "r--", lw=1.5, label="prediction")
    ax.set_title(f"{title} — test res {resolution_test}")
    ax.legend()
plt.tight_layout()
plt.show()
