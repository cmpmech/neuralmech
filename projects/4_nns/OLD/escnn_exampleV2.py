import time

import escnn
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn
from tqdm import tqdm

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------- settings ---------------------------
resolution = 128
epochs = 2000
lr = 5e-3
cost_fun = nn.MSELoss()

# SO(2) acting on R^2 (trivial scalar fields)
# gspace = gspaces.rot2dOnR2(N=8)  # discrete C8 rotation group (cheaper than full SO2)
gspace = gspaces.rot2dOnR2(N=12)  # discrete C8 rotation group (cheaper than full SO2)
# gspace = gspaces.rot2dOnR2(N=-1)  # continuous TODO DOES NOT WORK NOW?
# gspace = gspaces.rot2dOnR2(N=4)
trivial = (
    gspace.trivial_repr
)  # scalar representation: irrep(1) for 2D vector), irrep(2) for second order tensor

# channel counts (number of field copies)
hidden_channels = 8

# ----------------------------- data -----------------------------
x1 = np.linspace(-1, 1, resolution)
x2 = np.linspace(-1, 1, resolution)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

# # input: simple sine (different from target)
# x_np = x1 * x2  # np.sin(4 * np.pi * x1 * x2)
# # target: product sine
# y_np = np.sin(2 * np.pi * x1) * np.sin(12 * np.pi * x1 * x2)

x_np = np.sin(4 * np.pi * x1 * x2)
y_np = np.cos(4 * np.pi * x1 * x2)  # normalized derivative

y = torch.from_numpy(y_np).float().to(device).unsqueeze(0).unsqueeze(0)

# wrap input as escnn GeometricTensor
x_tensor = torch.from_numpy(x_np).float().to(device).unsqueeze(0).unsqueeze(0)

# inscribed disk: only here is the square grid actually C8-equivariant
mask = torch.from_numpy((x1**2 + x2**2) <= 1.0).to(device)
# # disable mask
# mask = mask * 0 + 1

# restrict the input support to the disk so it is rotation-consistent
# x_tensor = x_tensor * mask


# ---------------------------- model ----------------------------
KERNEL_SIZE = 9


class EquivariantCNN(nn.Module):
    def __init__(self, gspace, hidden_channels):
        super().__init__()
        trivial = gspace.trivial_repr
        regular = gspace.regular_repr

        in_type = enn.FieldType(gspace, [trivial])
        h_type = enn.FieldType(gspace, [regular] * hidden_channels)
        out_type = enn.FieldType(gspace, [trivial])

        self.net = enn.SequentialModule(
            enn.R2Conv(
                in_type,
                h_type,
                kernel_size=KERNEL_SIZE,
                padding=KERNEL_SIZE // 2,
                bias=False,
            ),
            # enn.ReLU(h_type),
            enn.LeakyReLU(h_type),
            # enn.ELU(h_type),
            enn.R2Conv(
                h_type,
                h_type,
                kernel_size=KERNEL_SIZE,
                padding=KERNEL_SIZE // 2,
                bias=False,
            ),
            # enn.ReLU(h_type),
            enn.LeakyReLU(h_type),
            # enn.ELU(h_type),
            enn.R2Conv(
                h_type,
                h_type,
                kernel_size=KERNEL_SIZE,
                padding=KERNEL_SIZE // 2,
                bias=False,
            ),
            # enn.ReLU(h_type),
            enn.LeakyReLU(h_type),
            # enn.ELU(h_type),
            enn.R2Conv(
                h_type,
                out_type,
                kernel_size=KERNEL_SIZE,
                padding=KERNEL_SIZE // 2,
                bias=False,
            ),
        )
        self.in_type = in_type

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        return self.net(x).tensor


model = EquivariantCNN(gspace, hidden_channels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)

# -------------------------- training ---------------------------
train_cost = []
pbar = tqdm(range(epochs))
model.train()
tic = time.time()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    cost = cost_fun(y_pred[0, 0][mask], y[0, 0][mask])
    # cost = cost_fun(
    #     y_pred[0, 0], y[0, 0]
    # )
    cost.backward()
    optimizer.step()
    train_cost.append(cost.item())
    if epoch % 10 == 0:
        pbar.set_postfix({"train": f"{cost.item():.2e}"})
print(f"elapsed time {time.time() - tic:.2f} s")

# ------------------------ postprocessing -----------------------
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(train_cost, "k")
plt.show()

mask_np = mask.cpu().numpy()
for data, name in [(y_pred, "prediction"), (y, "target"), (x_tensor, "input")]:
    field = np.where(mask_np, data[0, 0].detach().cpu().numpy(), np.nan)
    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax.pcolormesh(x1, x2, field, vmin=-1, vmax=1, cmap="Spectral")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(f"../../results/ESCNN_{name}.png", bbox_inches="tight", pad_inches=0)
    plt.show()

# ------------------- rotated input (30 degrees) ----------------
x_rot_np = scipy.ndimage.rotate(x_np, 30, reshape=False, order=1, mode="reflect")
x_rot_tensor = torch.from_numpy(x_rot_np).float().to(device).unsqueeze(0).unsqueeze(0)
x_rot_tensor = x_rot_tensor * mask

model.eval()
with torch.no_grad():
    y_pred_rot = model(x_rot_tensor)

for data, name in [(y_pred_rot, "prediction_rot30"), (x_rot_tensor, "input_rot30")]:
    field = np.where(mask_np, data[0, 0].detach().cpu().numpy(), np.nan)
    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax.pcolormesh(x1, x2, field, vmin=-1, vmax=1, cmap="Spectral")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(f"../../results/ESCNN_{name}.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    gspace = gspaces.rot2dOnR2(
        N=8
    )  # discrete C8 rotation group (cheaper than full SO2)
