import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------- settings ---------------------------
RESOLUTION = 128
# RESOLUTION = 256
EPOCHS = 2000
LR = 5e-3
KERNEL_SIZE = 7
HIDDEN_CHANNELS = 6
FREQUENCY = 2 * np.pi  # low frequency -> bandlimited -> rotation does not alias
N = 8  # discrete rotation group C_N (group angles are multiples of 360/N)
CHECK_ANGLES = [0, 15, 22.5, 30, 45, 90]  # incl. angles NOT in the group

gspace = gspaces.rot2dOnR2(N=N)

# ----------------------------- data -----------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

x_np = np.sin(FREQUENCY * (x1 + 0.5) * (x2 + 0.5))
y_np = np.cos(FREQUENCY * (x1 + 0.5) * (x2 + 0.5))

y = torch.from_numpy(y_np).float().to(device).unsqueeze(0).unsqueeze(0)
x_tensor = torch.from_numpy(x_np).float().to(device).unsqueeze(0).unsqueeze(0)

# inscribed disk: the only region a square grid can be rotated without losing support
mask_np = (x1**2 + x2**2) <= 1.0
mask = torch.from_numpy(mask_np).to(device)

rotate = lambda field, angle: scipy.ndimage.rotate(
    field, angle, reshape=False, order=1, mode="reflect"
)


# ---------------------------- model ----------------------------
class EquivariantCNN(nn.Module):
    def __init__(self, gspace, hidden_channels):
        super().__init__()
        trivial = gspace.trivial_repr
        regular = gspace.regular_repr

        in_type = enn.FieldType(gspace, [trivial])
        h_type = enn.FieldType(gspace, [regular] * hidden_channels)
        out_type = enn.FieldType(gspace, [trivial])

        pad = KERNEL_SIZE // 2
        self.net = enn.SequentialModule(
            enn.MaskModule(in_type, RESOLUTION, margin=1.0),  # kill corner support
            enn.R2Conv(
                in_type, h_type, kernel_size=KERNEL_SIZE, padding=pad, bias=False
            ),
            enn.LeakyReLU(h_type),
            enn.R2Conv(
                h_type, h_type, kernel_size=KERNEL_SIZE, padding=pad, bias=False
            ),
            enn.LeakyReLU(h_type),
            enn.R2Conv(
                h_type, h_type, kernel_size=KERNEL_SIZE, padding=pad, bias=False
            ),
            enn.LeakyReLU(h_type),
            enn.R2Conv(
                h_type, out_type, kernel_size=KERNEL_SIZE, padding=pad, bias=False
            ),
        )
        self.in_type = in_type

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        return self.net(x).tensor


model = EquivariantCNN(gspace, HIDDEN_CHANNELS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)

# -------------------------- training ---------------------------
train_cost = []
cost_fun = nn.MSELoss()
pbar = tqdm(range(EPOCHS))
model.train()
tic = time.time()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    cost = cost_fun(y_pred[0, 0][mask], y[0, 0][mask])
    cost.backward()
    optimizer.step()
    train_cost.append(cost.item())
    if epoch % 10 == 0:
        pbar.set_postfix({"train": f"{cost.item():.2e}"})
print(f"elapsed time {time.time() - tic:.2f} s")

# ----------------------- equivariance check --------------------
# compare f(R x) against R f(x): equivariance holds when these agree
model.eval()
with torch.no_grad():
    pred_np = model(x_tensor)[0, 0].cpu().numpy()
for angle in CHECK_ANGLES:
    x_rot = torch.from_numpy(rotate(x_np, angle)).float().to(device)[None, None]
    with torch.no_grad():
        f_of_rot = model(x_rot)[0, 0].cpu().numpy()
    rot_of_f = rotate(pred_np, angle)
    err = np.sqrt(np.mean((f_of_rot - rot_of_f)[mask_np] ** 2))
    print(f"angle {angle:5.1f} deg : equivariance rmse {err:.3e}")

# ------------------------ postprocessing -----------------------
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(train_cost, "k")
plt.show()

ANGLE = 90
x_rot45 = torch.from_numpy(rotate(x_np, ANGLE)).float().to(device)[None, None]
with torch.no_grad():
    f_of_rot45 = model(x_rot45)[0, 0].cpu().numpy()
rot45_of_f = rotate(pred_np, ANGLE)

panels = [
    (pred_np, "prediction"),
    (y_np, "target"),
    (x_np, "input"),
    (f_of_rot45, "prediction_Rx45"),  # f(R x): rotate input, then predict
    (rot45_of_f, "prediction_fx45"),  # R f(x): predict, then rotate (equivariance ref)
]
for field, name in panels:
    field = np.where(mask_np, field, np.nan)
    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.pcolormesh(x1, x2, field, vmin=-1, vmax=1, cmap="Spectral")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(RESULTS_DIR / f"ESCNN_V3_{name}.png", bbox_inches="tight", pad_inches=0)
    plt.show()
