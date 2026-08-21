import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from optimization_config import ackley as objective

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
f, xrange, yrange, guess = (
    objective.f,
    objective.xrange,
    objective.yrange,
    objective.guess,
)
EPOCHS = 200
LR = 0.4
RESOLUTION = 800


# ------------------------------------ optimization -----------------------------------
def f_torch(x):  # objective.f is numpy only, mirror it for autograd
    r = torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    c = 0.5 * (torch.cos(2 * torch.pi * x[0]) + torch.cos(2 * torch.pi * x[1]))
    return -20 * torch.exp(-0.2 * r) - torch.exp(c) + 20 + torch.e


x = torch.nn.Parameter(torch.tensor(guess))
optimizer = torch.optim.Adam([x], lr=LR)

trajectory = [x.data.clone()]
for _ in range(EPOCHS):
    optimizer.zero_grad()
    f_torch(x).backward()
    optimizer.step()
    trajectory.append(x.data.clone())
trajectory = torch.stack(trajectory)

print(f"final: x={trajectory[-1, 0]:.2e}, y={trajectory[-1, 1]:.2e}")

# ----------------------------------- postprocessing ----------------------------------
x_grid = np.linspace(*xrange, RESOLUTION)
y_grid = np.linspace(*yrange, RESOLUTION)
xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
z = f(np.stack([xx, yy], axis=0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=RESOLUTION // 4)
ax.contourf(xx, yy, z, levels=48, cmap="cividis")
ax.plot(trajectory[:, 0], trajectory[:, 1], "k", linewidth=3)
ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ko", ms=4)
ax.plot(0, 0, "ws", ms=4)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)  # avoid contourline artifacts
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig.savefig(RGB_PDF_DIR / "ackley_adam.pdf")
