import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from escnn import gspaces
from escnn import nn as enn
from torch import nn
from tqdm import tqdm

from NN import EquivariantCNN

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 65
EPOCHS = 500
LR = 1e-2

# define loss
cost_fun = nn.MSELoss()

# model settings
ROTATIONS = 1  # standard cnn recovered
# ROTATIONS = 8  # discrete C8 rotation group acting on R^2

# input carries 2 vector fields, output 1; channels are field copies
CHANNELS = [2] + [32 // ROTATIONS] * 3 + [1]
KERNEL_SIZE = 7
PADDING = KERNEL_SIZE // 2
ACTIVATION = enn.LeakyReLU

# postprocessing
ANGLES = [0, 30, 45, 90]
STRIDE = 8  # quiver subsampling stride

# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

r2 = x1**2 + x2**2
ones = np.ones_like(x1)
zeros = np.zeros_like(x1)

# input: coordinate field (x1, x2) -- invariant under the joint group action --
# stacked with a constant load direction d = (1, 0) that breaks the symmetry.
# target: the equivariant function f = (x . d) d = (0.5 x1 + 1, 0)
x_np = np.stack([x1, x2, ones, zeros], axis=0)
y_np = np.stack([0.5 * x1 + 1, zeros], axis=0)

x = torch.from_numpy(x_np).to(torch.float32).to(device).unsqueeze(0)
y = torch.from_numpy(y_np).to(torch.float32).to(device).unsqueeze(0)

# only within the inscribed disk is the square grid actually rotation-equivariant
mask = torch.from_numpy(r2 <= 1.0).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
gspace = gspaces.rot2dOnR2(N=ROTATIONS)
# the 2D vector lives in the frequency-1 irrep; only C_N with N > 2 carries it, so the
# standard CNN (ROTATIONS=1) treats the two components as independent scalars
if ROTATIONS > 1:
    vector_repr = gspace.irrep(1)
else:
    vector_repr = gspace.trivial_repr + gspace.trivial_repr

model = EquivariantCNN(
    gspace,
    CHANNELS,
    ACTIVATION,
    kernel_size=KERNEL_SIZE,
    padding=PADDING,
    in_repr=vector_repr,
    out_repr=vector_repr,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 50
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x)
    cost = cost_fun(y_pred[0][:, mask], y[0][:, mask])
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
mask_np = mask.cpu().numpy()
model.eval()


# rotate a stack of 2D vector fields by the joint group action: each field is
# rotated in space and its components mixed by the 2x2 rotation matrix
def rotate_fields(field_np, angle):
    rad = np.deg2rad(angle)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    out = np.empty_like(field_np)
    for k in range(field_np.shape[0] // 2):
        f1 = scipy.ndimage.rotate(field_np[2 * k], angle, reshape=False)
        f2 = scipy.ndimage.rotate(field_np[2 * k + 1], angle, reshape=False)
        out[2 * k] = R[0, 0] * f1 + R[0, 1] * f2
        out[2 * k + 1] = R[1, 0] * f1 + R[1, 1] * f2
    return out


targets, preds = {}, {}
for angle in ANGLES:
    x_rot = torch.from_numpy(rotate_fields(x_np, angle)).to(torch.float32)
    with torch.no_grad():
        pred = model(x_rot.to(device).unsqueeze(0))[0].cpu().numpy()
    target = rotate_fields(y_np, angle)
    targets[angle] = target
    preds[angle] = pred

sub = (slice(None, None, STRIDE), slice(None, None, STRIDE))
xs, ys = x1[sub], x2[sub]

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    for angle in ANGLES:
        for field in (targets[angle], preds[angle]):
            fig, ax = plt.subplots()
            mag = np.sqrt(field[0] ** 2 + field[1] ** 2)

            ax.contourf(x1, x2, mag, cmap="cividis", levels=64, vmin=0, vmax=1.5)

            field[:, ~mask_np] = np.nan
            # ax.quiver(xs, ys, field[0][sub], field[1][sub], pivot="mid", width=0.005)
            ax.streamplot(
                x1[:, 0],
                x2[0, :],
                field[0],
                field[1],
                color="k",
                density=1,
                linewidth=2,
                arrowsize=0,
            )
            ax.set_aspect("equal")
            ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for angle in ANGLES:
        for name, field in (("target", targets[angle]), ("prediction", preds[angle])):
            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            mag = np.sqrt(field[0] ** 2 + field[1] ** 2)
            ax.contourf(x1, x2, mag, cmap="cividis", levels=64, vmin=0, vmax=1.5)
            field[:, ~mask_np] = np.nan
            ax.streamplot(
                x1[:, 0],
                x2[0, :],
                field[0],
                field[1],
                color="k",
                density=0.8,
                linewidth=2,
                arrowsize=0,
            )

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_rasterized(True)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(RGB_PDF_DIR / f"ESCNN_vector_{name}_{angle}_{ROTATIONS}.pdf")
            plt.close()
