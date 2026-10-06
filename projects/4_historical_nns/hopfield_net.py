import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
CORRUPTION = 0.3  # fraction of flipped pixels
SWEEPS = 10

# model settings
LETTERS = {
    "F": [
        ".########.",
        ".########.",
        ".##.......",
        ".##.......",
        ".#######..",
        ".#######..",
        ".##.......",
        ".##.......",
        ".##.......",
        "..........",
    ],
    "W": [
        "##......##",
        "##......##",
        "##......##",
        "##......##",
        "##..##..##",
        "##..##..##",
        "##.####.##",
        ".###..###.",
        ".##....##.",
        "..........",
    ],
    "S": [
        "..######..",
        ".##....##.",
        ".##.......",
        "..###.....",
        "...####...",
        ".....###..",
        ".......##.",
        ".##....##.",
        "..######..",
        "..........",
    ],
}


# -------------------------------------- helper ---------------------------------------
def store(X):
    W = X.T @ X / X.shape[0]
    W.fill_diagonal_(0)
    return W


def recall(W, s, sweeps=10):
    for _ in range(sweeps):
        for i in torch.randperm(len(s)):
            s[i] = 1 if s @ W[:, i] >= 0 else -1
    return s


energy = lambda W, s: -0.5 * s @ W @ s

# ------------------------------------ create data ------------------------------------
X = torch.stack(
    [
        torch.tensor([[1.0 if c == "#" else -1.0 for c in row] for row in letter]).flatten()
        for letter in LETTERS.values()
    ]
).to(device)
RESOLUTION = len(LETTERS["F"])

flip = torch.rand(X.shape, device=device) < CORRUPTION
X_corrupted = torch.where(flip, -X, X)

# -------------------------------------- training -------------------------------------
W = store(X)

# ----------------------------------- postprocessing ----------------------------------
X_recalled = torch.stack([recall(W, s.clone(), SWEEPS) for s in X_corrupted])

for name, x, s_corrupted, s_recalled in zip(LETTERS, X, X_corrupted, X_recalled):
    print(
        f"{name}: energy corrupted {energy(W, s_corrupted):.2e}, recalled {energy(W, s_recalled):.2e}, "
        f"pixel errors corrupted {int((s_corrupted != x).sum())}, recalled {int((s_recalled != x).sum())}"
    )

stages = {"stored": X, "corrupted": X_corrupted, "recalled": X_recalled}

if not args.book:
    fig, axes = plt.subplots(len(LETTERS), len(stages), figsize=(len(stages), len(LETTERS)))
    for row, idx in zip(axes, range(len(LETTERS))):
        for ax, images in zip(row, stages.values()):
            ax.imshow(images[idx].view(RESOLUTION, RESOLUTION).cpu(), cmap="binary")
            ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for idx, name in enumerate(LETTERS):
        for stage, images in stages.items():
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(images[idx].view(RESOLUTION, RESOLUTION).cpu(), cmap="binary")
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(RGB_PDF_DIR / f"hopfield_{name}_{stage}.pdf")
            plt.close(fig)
