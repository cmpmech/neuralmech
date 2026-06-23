import argparse
import time
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from NN import MLP
from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Init(Enum):
    WIDE_UNIFORM = 0
    UNIT_UNIFORM = 1
    XAVIER_TANH = 2


# -------------------------------------- settings -------------------------------------
# weight initialization
INITIALIZATION = Init.WIDE_UNIFORM
# INITIALIZATION = Init.UNIT_UNIFORM
# INITIALIZATION = Init.XAVIER_TANH

# hyperparameters
HIDDEN_LAYERS = 4
NEURONS = 128
RESOLUTION = 800
activation = torch.nn.Tanh

cost_fun = nn.MSELoss(reduction="mean")

# --------------------------------------- setup ---------------------------------------
if args.book or args.animate:
    close = True
    if args.book:
        save_at = [0, 10, 100, 200, 400, 1000, 10000]
    else:
        SAVE_EVERY = 1
        save_at = np.arange(0, 2000, SAVE_EVERY)
        folder = f"learning_image_{INITIALIZATION}"
        (ANIMATION_DIR / folder).mkdir(parents=True, exist_ok=True)
else:
    close = False
    save_at = []


# --------------------------------------- helper --------------------------------------
def init_weights(m):
    if type(m) == torch.nn.Linear:
        if INITIALIZATION == Init.WIDE_UNIFORM:
            torch.nn.init.uniform_(m.weight, a=-10, b=10)
            torch.nn.init.uniform_(m.bias, a=-10, b=10)
        elif INITIALIZATION == Init.UNIT_UNIFORM:
            torch.nn.init.uniform_(m.weight, a=-1, b=1)
            torch.nn.init.uniform_(m.bias, a=-1, b=1)
        elif INITIALIZATION == Init.XAVIER_TANH:
            gain = nn.init.calculate_gain("tanh")
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)


def normalize_output(y, eps=1e-8):
    y_min = y.amin(dim=(0, 1), keepdim=True)
    y_max = y.amax(dim=(0, 1), keepdim=True)
    return (y - y_min) / (y_max - y_min + eps)


# ----------------------------------- preprocessing -----------------------------------
layers = [2] + [NEURONS] * HIDDEN_LAYERS + [3]
activations = [activation() for _ in range(HIDDEN_LAYERS)]

x = torch.linspace(-1, 1, RESOLUTION)
y = torch.linspace(-1, 1, RESOLUTION)
x, y = torch.meshgrid(x, y, indexing="ij")
mlp_input = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(layers, post_modules=activations)
model.apply(init_weights)
model.to(device)


if INITIALIZATION == Init.WIDE_UNIFORM:
    LR, EPOCHS = 2e-2, 10001
elif INITIALIZATION == Init.UNIT_UNIFORM:
    LR, EPOCHS = 1e-2, 10001
elif INITIALIZATION == Init.XAVIER_TANH:
    LR, EPOCHS = 2e-3, 10001
if args.animate:
    EPOCHS = EPOCHS // 5

optimizer = torch.optim.AdamW(model.parameters(), LR)

# ------------------------------------- load data -------------------------------------
img = Image.open(DATA_DIR / "images/memphis.jpg").convert("RGB")
target = torch.from_numpy(np.asarray(img)).to(torch.float32).to(device)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 1
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(mlp_input).reshape(RESOLUTION, RESOLUTION, 3)
    cost = cost_fun(y_pred, target)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})

# ----------------------------------- postprocessing ----------------------------------
    if epoch in save_at:
        if args.book:
            path = RESULTS_DIR / f"learning_image_{INITIALIZATION}_{epoch}.png"
        elif args.animate:
            path = ANIMATION_DIR / f"{folder}/frame_{epoch // SAVE_EVERY}.jpg"
        else:
            path = None
        show_image(
            normalize_output(y_pred.detach().cpu()).numpy(), path=path, close=close
        )

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

with torch.no_grad():
    z_pred = model(mlp_input).reshape(RESOLUTION, RESOLUTION, 3).cpu()
    z_pred = normalize_output(z_pred)

# ----------------------------------- postprocessing ----------------------------------
if not args.book and not args.animate:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

if not args.animate:
    show_image(
        normalize_output(target.detach().cpu()).numpy(),
        path=RESULTS_DIR / "memphis_target.png",
        close=close,
    )
