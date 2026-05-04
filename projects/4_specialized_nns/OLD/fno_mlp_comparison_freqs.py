import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralop.models import FNO
from torch import nn
from tqdm import tqdm

from NN import DCN

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- training settings ---------------------------
resolution_train = 256
resolution_test = 1024
n_samples = 200
freq_min = 2.0
freq_max = 20.0
phi = 0.5 * np.pi
epochs = 300
lr = 1e-3
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
# FNO
k_modes = 16
hidden_channels = 32
n_layers = 2

# CNN — fully convolutional, no dense layers, so resolution-flexible
cnn_channels = [1, 32, 32, 32, 1]
cnn_activations = [nn.GELU(approximate="tanh")] * (len(cnn_channels) - 2) + [None]
kernel_size = 31  # large kernel to get decent receptive field
padding = kernel_size // 2


# ----------------------------- prepare data -----------------------------
def make_data(resolution, n_samples, freq_min, freq_max, phi, device):
    z = np.linspace(-1, 1, resolution)
    freqs = np.random.uniform(freq_min, freq_max, n_samples)
    # freqs = np.linspace(freq_min, freq_max, n_samples)
    x = np.stack([np.sin(f * z) for f in freqs], axis=0)
    y = np.stack([np.sin(f * (z + phi)) for f in freqs], axis=0)

    # FNO and CNN both expect (batch, channels, n_points)
    x_t = torch.from_numpy(x).to(torch.float32).unsqueeze(1).to(device)
    y_t = torch.from_numpy(y).to(torch.float32).unsqueeze(1).to(device)

    return z, x_t, y_t


z_train, x_train, y_train = make_data(
    resolution_train, n_samples, freq_min, freq_max, phi, device
)
z_test, x_test, y_test = make_data(
    resolution_test, n_samples, freq_min, freq_max, phi, device
)

print(z_train.shape, x_train.shape, y_train.shape)

# # -------------------- instantiate models & optimizers -------------------
# fno = FNO(
#     n_modes=(k_modes,),
#     in_channels=1,
#     out_channels=1,
#     hidden_channels=hidden_channels,
#     n_layers=n_layers,
# ).to(device)

# cnn = DCN(
#     channels=cnn_channels,
#     activations=cnn_activations,
#     kernel_size=kernel_size,
#     stride=1,
#     padding=padding,
#     dim=1,
# ).to(device)

# optimizer_fno = torch.optim.Adam(fno.parameters(), lr=lr)
# optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=lr)


# # ------------------------------- training -------------------------------
# def train(model, optimizer, x, y, epochs, label):
#     costs = []
#     pbar = tqdm(range(epochs), desc=label)
#     model.train()
#     tic = time.time()
#     for epoch in pbar:
#         optimizer.zero_grad()
#         y_pred = model(x)
#         cost = cost_fun(y_pred, y)
#         cost.backward()
#         optimizer.step()
#         costs.append(cost.item())
#         if epoch % 50 == 0:
#             pbar.set_postfix({"loss": f"{cost.item():.2e}"})
#     print(f"{label} elapsed: {time.time() - tic:.2f} s")
#     return costs


# costs_fno = train(fno, optimizer_fno, x_train, y_train, epochs, "FNO")
# costs_cnn = train(cnn, optimizer_cnn, x_train, y_train, epochs, "CNN")

# # ---------------------------- postprocessing ----------------------------
# fig, ax = plt.subplots()
# ax.set_yscale("log")
# ax.plot(costs_fno, "b", label="FNO")
# ax.plot(costs_cnn, "r", label="CNN")
# ax.legend()
# ax.set_xlabel("epoch")
# ax.set_ylabel("MSE")
# plt.show()

# fno.eval()
# cnn.eval()

# # predictions at training resolution
# with torch.no_grad():
#     y_pred_fno_train = fno(x_train).squeeze(1).cpu()
#     y_pred_cnn_train = cnn(x_train).squeeze(1).cpu()

# n_show = 3
# fig, axes = plt.subplots(2, n_show, figsize=(12, 6), dpi=150)
# for i in range(n_show):
#     for row, (pred, title) in enumerate(
#         [
#             (y_pred_fno_train, "FNO"),
#             (y_pred_cnn_train, "CNN"),
#         ]
#     ):
#         axes[row, i].plot(z_train, y_train[i, 0].cpu(), "k", lw=1.5, label="target")
#         axes[row, i].plot(z_train, pred[i], "r--", lw=1.5, label="prediction")
#         axes[row, i].set_title(f"{title} sample {i + 1} — res {resolution_train}")
#         axes[row, i].legend(fontsize=7)
# plt.tight_layout()
# plt.show()

# # resolution generalization
# with torch.no_grad():
#     y_pred_fno_test = fno(x_test).squeeze(1).cpu()
#     y_pred_cnn_test = cnn(x_test).squeeze(1).cpu()

# fig, axes = plt.subplots(2, n_show, figsize=(12, 6), dpi=150)
# for i in range(n_show):
#     for row, (pred, title) in enumerate(
#         [
#             (y_pred_fno_test, "FNO"),
#             (y_pred_cnn_test, "CNN"),
#         ]
#     ):
#         axes[row, i].plot(z_test, y_test[i, 0].cpu(), "k", lw=1.5, label="target")
#         axes[row, i].plot(z_test, pred[i], "r--", lw=1.5, label="prediction")
#         axes[row, i].set_title(f"{title} sample {i + 1} — res {resolution_test}")
#         axes[row, i].legend(fontsize=7)
# plt.suptitle(
#     f"Resolution generalization: trained on {resolution_train}, tested on {resolution_test}"
# )
# plt.tight_layout()
# plt.show()
