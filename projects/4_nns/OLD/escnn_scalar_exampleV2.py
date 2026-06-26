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


# ---------------------------------------------------------------------------
# Continuous-group network. Add this to NN.py next to EquivariantCNN, or keep
# it here. For the continuous group SO(2) (gspace built with N=-1) the regular
# representation is infinite-dimensional, so hidden fields cannot use it.
# Instead, hidden fields are band-limited Fourier fields (a direct sum of
# irreps up to max_freq) and the nonlinearity is a Fourier nonlinearity: it
# samples the group at n_samples points, applies a pointwise ELU, and maps
# back. Input and output default to scalar (trivial) fields.
# ---------------------------------------------------------------------------
class EquivariantCNNContinuous(nn.Module):
    def __init__(
        self,
        gspace,
        channels: list[int],
        kernel_size: int = 7,
        padding: int = 0,
        max_freq: int = 3,
        n_samples: int = 16,
        in_repr=None,
        out_repr=None,
    ) -> None:
        super().__init__()
        in_repr = in_repr if in_repr is not None else gspace.trivial_repr
        out_repr = out_repr if out_repr is not None else gspace.trivial_repr

        # band-limited irreps up to frequency max_freq, e.g. [(0,), (1,), (2,), (3,)]
        irreps = [(f,) for f in range(max_freq + 1)]

        in_type = enn.FieldType(gspace, [in_repr] * channels[0])
        modules = []
        current = in_type
        for i in range(1, len(channels) - 1):
            # the Fourier nonlinearity defines the hidden field type
            act = enn.FourierELU(
                gspace, channels[i], irreps, N=n_samples, type="regular"
            )
            modules.append(
                enn.R2Conv(current, act.in_type, kernel_size, padding=padding)
            )
            modules.append(act)
            current = act.out_type
        out_type = enn.FieldType(gspace, [out_repr] * channels[-1])
        modules.append(enn.R2Conv(current, out_type, kernel_size, padding=padding))

        self.in_type = in_type
        self.model = enn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(enn.GeometricTensor(x, self.in_type)).tensor


BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 128
EPOCHS = 1000
LR = 2e-3

# define loss
cost_fun = nn.MSELoss()

# model settings
# ROTATIONS = 1   # standard CNN recovered
# ROTATIONS = 8   # discrete C8 rotation group (cheaper than full SO(2))
ROTATIONS = -1  # continuous SO(2)

# continuous-group settings (only used when ROTATIONS == -1)
MAX_FREQ = 3  # band limit for the hidden irrep fields
N_SAMPLES = 16  # group samples for the Fourier nonlinearity (not the group order)

if ROTATIONS != -1:
    CHANNELS = [1] + [64 // ROTATIONS] * 4 + [1]
else:
    CHANNELS = [1] + [8] * 4 + [1]
KERNEL_SIZE = 7
PADDING = KERNEL_SIZE // 2
ACTIVATION = enn.LeakyReLU  # used only in the discrete case

# postprocessing
ANGLES = [0, 30, 45, 90]

# ------------------------------------ prepare data -----------------------------------
x1 = np.linspace(-1, 1, RESOLUTION)
x2 = np.linspace(-1, 1, RESOLUTION)
x1, x2 = np.meshgrid(x1, x2, indexing="ij")

x_np = np.sin(4 * np.pi * x1 * x2)
y_np = np.cos(4 * np.pi * x1 * x2)
x = torch.from_numpy(x_np).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
y = torch.from_numpy(y_np).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)

# only within the inscribed disk is the square grid actually rotation-equivariant
mask = torch.from_numpy((x1**2 + x2**2) <= 2.0).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
gspace = gspaces.rot2dOnR2(N=ROTATIONS)
if ROTATIONS == -1:
    model = EquivariantCNNContinuous(
        gspace,
        CHANNELS,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        max_freq=MAX_FREQ,
        n_samples=N_SAMPLES,
    ).to(device)
else:
    model = EquivariantCNN(
        gspace, CHANNELS, ACTIVATION, kernel_size=KERNEL_SIZE, padding=PADDING
    ).to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 10
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(x)
    cost = cost_fun(y_pred[0, 0][mask], y[0, 0][mask])
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
mask_np = mask.cpu().numpy()


def rotate(field_np, angle):
    rotated = scipy.ndimage.rotate(
        field_np, angle, reshape=False, order=1, mode="reflect"
    )
    return (
        torch.from_numpy(rotated).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    )


model.eval()
fields = {}
for angle in ANGLES:
    with torch.no_grad():
        y_pred_rot = model(rotate(x_np, angle))
    fields[f"target_{angle}"] = rotate(y_np, angle)
    fields[f"prediction_{angle}"] = y_pred_rot


def masked(field):
    return np.where(mask_np, field[0, 0].detach().cpu().numpy(), np.nan)


if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

    for field in fields.values():
        fig, ax = plt.subplots()
        ax.pcolormesh(x1, x2, masked(field), vmin=-1, vmax=1, cmap="Spectral")
        ax.set_aspect("equal")
        ax.axis("off")
        plt.show()
# -------------------------------- book postprocessing --------------------------------
else:

    def save_field(field, path):
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
        ax.pcolormesh(x1, x2, masked(field), vmin=-1, vmax=1, cmap="Spectral")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path)
        plt.close()

    for name, field in fields.items():
        save_field(field, RESULTS_DIR / f"ESCNN_{name}_{ROTATIONS}.png")
