import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 500  # full-batch gradient descent steps
N = 16
D = 16

# sweep
RESOLUTION = 512
LR_RANGE = (1e-1, 1e2)  # learning rates of the first and second layer
ZOOMS = 4
ZOOM_FACTOR = 8
DIVERGENCE = 1e6

# ------------------------------------ create data ------------------------------------
X = torch.randn(N, D, device=device)
Y = torch.randn(N, device=device)
W1_init = torch.randn(D, D, device=device) / np.sqrt(D)
w2_init = torch.randn(D, device=device) / np.sqrt(D)


# -------------------------------------- helper ---------------------------------------
def train_grid(log_lr1, log_lr2):
    lr1 = torch.from_numpy(10**log_lr1).to(device, torch.float32).flatten()
    lr2 = torch.from_numpy(10**log_lr2).to(device, torch.float32).flatten()
    G = len(lr1)
    W1 = W1_init.expand(G, D, D).clone()
    w2 = w2_init.expand(G, D).clone()
    cost_sum = torch.zeros(G, device=device)
    diverged_at = torch.full((G,), EPOCHS, device=device)
    for epoch in range(EPOCHS):
        h = torch.tanh(torch.einsum("nd,gkd->gnk", X, W1) / np.sqrt(D))
        r = torch.einsum("gnk,gk->gn", h, w2) / np.sqrt(D) - Y
        cost = (r**2).mean(dim=1)
        diverged = ~torch.isfinite(cost) | (cost > DIVERGENCE)
        diverged_at = torch.where(diverged & (diverged_at == EPOCHS), epoch, diverged_at)
        cost_sum += torch.where(diverged, 0.0, cost)
        grad_w2 = 2 / N * torch.einsum("gnk,gn->gk", h, r) / np.sqrt(D)
        delta = 2 / N * r[:, :, None] * w2[:, None, :] * (1 - h**2) / np.sqrt(D)
        grad_W1 = torch.einsum("gnk,nd->gkd", delta, X) / np.sqrt(D)
        W1 = torch.where(diverged[:, None, None], W1, W1 - lr1[:, None, None] * grad_W1)
        w2 = torch.where(diverged[:, None], w2, w2 - lr2[:, None] * grad_w2)

    # converged runs negative (mean cost), diverged runs positive (speed of divergence)
    converged = diverged_at == EPOCHS
    mean_cost = torch.log10(cost_sum / EPOCHS)
    low, high = mean_cost[converged].min(), mean_cost[converged].max()
    mean_cost = (mean_cost - low) / (high - low)
    speed = 1 - torch.log(diverged_at + 1) / np.log(EPOCHS + 1)
    field = torch.where(converged, -1 + 0.9 * mean_cost, 0.1 + 0.9 * speed / speed.max())
    return field.reshape(RESOLUTION, RESOLUTION).cpu().numpy()


def find_boundary(field, log_lr1, log_lr2):
    # zoom into the central boundary point with the most intertwined converged/diverged pixels
    edge = np.zeros_like(field, dtype=float)
    edge[:, 1:] = (field[:, 1:] > 0) != (field[:, :-1] > 0)
    mixing = uniform_filter(edge, size=RESOLUTION // 16)
    quarter = RESOLUTION // 4
    mixing[:quarter], mixing[-quarter:], mixing[:, :quarter], mixing[:, -quarter:] = 0, 0, 0, 0
    i, j = np.unravel_index(np.argmax(mixing * edge), field.shape)
    return log_lr1[i, j], log_lr2[i, j]


# ---------------------------------------- sweep --------------------------------------
lo, hi = np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1])
center1, center2, width = (lo + hi) / 2, (lo + hi) / 2, hi - lo
fields = []
tic = time.time()
for zoom in range(ZOOMS):
    axis1 = np.linspace(center1 - width / 2, center1 + width / 2, RESOLUTION)
    axis2 = np.linspace(center2 - width / 2, center2 + width / 2, RESOLUTION)
    log_lr2, log_lr1 = np.meshgrid(axis2, axis1, indexing="ij")
    field = train_grid(log_lr1, log_lr2)
    fields.append((field, axis1, axis2))
    print(f"zoom {zoom}: lr1 10^{center1:.4f}, lr2 10^{center2:.4f}, width {width:.1e}")
    center1, center2 = find_boundary(field, log_lr1, log_lr2)
    width /= ZOOM_FACTOR
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
figs = []
for field, axis1, axis2 in fields:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.imshow(field, cmap="cividis", vmin=-1, vmax=1, origin="lower",
              extent=[axis1[0], axis1[-1], axis2[0], axis2[-1]])
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figs.append(fig)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for zoom, fig in enumerate(figs):
        fig.savefig(RGB_PDF_DIR / f"trainability_fractal_{zoom}.pdf")
    extents = np.array([[axis1[0], axis1[-1], axis2[0], axis2[-1]] for _, axis1, axis2 in fields])
    save_csv(CSV_DIR / "trainability_fractal_extents.csv", zoom=np.arange(ZOOMS),
             lr1_min=extents[:, 0], lr1_max=extents[:, 1], lr2_min=extents[:, 2],
             lr2_max=extents[:, 3])
plt.close("all")
