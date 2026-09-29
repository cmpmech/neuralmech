import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import init_weights
from NN import MLP

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 128

# define loss
cost_fun = nn.CrossEntropyLoss()

# model settings
LAYERS = [3 * 28 * 28, 256, 256, 10]
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]

# shortcut
CORRELATION = 1.0  # probability that a training digit carries the color of its class, 0.99 still leaks the shape
SAMPLES = 10
COLORS = torch.tensor(plt.get_cmap("tab10").colors, dtype=torch.float32)

# ------------------------------------- load data -------------------------------------
rng = np.random.default_rng(0)
mnist = load_dataset("ylecun/mnist").with_format("numpy")


def colorize(split, correlation):
    X = torch.from_numpy(mnist[split][:]["image"]).float() / 255  # (N, 28, 28)
    Y = torch.from_numpy(mnist[split][:]["label"]).long()
    random_color = torch.from_numpy(rng.integers(0, 10, len(Y)))
    keep = torch.from_numpy(rng.random(len(Y)) < correlation)
    color_id = torch.where(keep, Y, random_color)
    X = X[:, None] * COLORS[color_id][:, :, None, None]  # (N, 3, 28, 28)
    return X, Y, color_id


X_train, Y_train, _ = colorize("train", CORRELATION)
X_test, Y_test, color_test = colorize("test", 0.0)  # colors carry no information at test
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE,
                          shuffle=True)

# --------------------------- instantiate model & optimizer ---------------------------
model = nn.Sequential(nn.Flatten(), MLP(LAYERS, ACTIVATIONS)).to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------------------- training --------------------------------------
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    train_cost = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        cost = cost_fun(model(x), y)
        cost.backward()
        optimizer.step()
        train_cost += cost.item()
    pbar.set_postfix({"train": f"{train_cost / len(train_loader):.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
model.eval()
with torch.no_grad():
    Y_train_pred = model(X_train.to(device)).argmax(dim=1).cpu()
    Y_test_pred = model(X_test.to(device)).argmax(dim=1).cpu()
print(f"train accuracy {100 * (Y_train_pred == Y_train).float().mean():.2f}%")
print(f"test accuracy {100 * (Y_test_pred == Y_test).float().mean():.2f}%")
print(f"test predictions that follow the color {100 * (Y_test_pred == color_test).float().mean():.2f}%")


def to_image(X):
    background = 1 - X.sum(dim=1, keepdim=True).clamp(0, 1)  # white, digits keep their color
    return (background + X).clamp(0, 1).permute(0, 2, 3, 1).numpy()


# one training sample per class, and the first test samples
train_ids = [int(torch.nonzero(Y_train == digit)[0]) for digit in range(10)]
strips = {"train": to_image(X_train[train_ids]), "test": to_image(X_test[:SAMPLES])}
print("test labels", Y_test[:SAMPLES].tolist())
print("test predictions", Y_test_pred[:SAMPLES].tolist())

figs = {}
for name, images in strips.items():
    fig, axs = plt.subplots(1, len(images), figsize=(len(images), 1))
    for ax, image in zip(axs, images):
        ax.imshow(image)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1)
    figs[name] = fig

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, fig in figs.items():
        fig.savefig(RGB_PDF_DIR / f"shortcut_mnist_{name}.pdf")
plt.close("all")
