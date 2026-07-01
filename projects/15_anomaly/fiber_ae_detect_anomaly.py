from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# ------------------------------ load data -------------------------------
domain_size = 256


data = []
for i in range(11):
    data.append(
        torch.from_numpy(
            np.load(BASE_DIR / f"../../data/fibers_anomaly_{i}_{domain_size}.npy")
        )
    )
    data[-1] = data[-1].to(torch.float32).unsqueeze(1).to(device)

# -------------------------- load trained model --------------------------
bottleneck_layers = 1

model = torch.load(
    BASE_DIR / f"../../models/fiber_ae_{bottleneck_layers}_{domain_size}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# --------------------------- reconstructions ----------------------------
reconstructions = []
errors = []
with torch.no_grad():
    for fibers in data:
        reconstructions.append(standardizex.inverse(model(standardizex(fibers))).cpu())
        errors.append((reconstructions[-1] - fibers.cpu()) ** 2)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
for i, error in enumerate(errors):
    ax.hist(torch.mean(error, dim=(1, 2, 3)), bins=50)
plt.show()


# ------------------------- book postprocessing --------------------------
anomaly_deg = 0  # 1 # 10
sample = 0

# reconstructions
fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
ax.imshow(
    data[anomaly_deg][sample][0].T.cpu(), origin="lower", cmap="binary", vmin=0, vmax=1
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/rgb_pdf/fibers_detection_true_{anomaly_deg}.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
ax.imshow(
    reconstructions[anomaly_deg][sample][0].T,
    origin="lower",
    cmap="binary",
    vmin=0,
    vmax=1,
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/rgb_pdf/fibers_detection_pred_{anomaly_deg}.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
ax.imshow(
    errors[anomaly_deg][sample][0].T,
    origin="lower",
    cmap="hot_r",
    norm=LogNorm(vmin=1e-5, vmax=1.3),
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/rgb_pdf/fibers_detection_error_{anomaly_deg}.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

# -------------------------------- export --------------------------------
for i, error in enumerate(errors):
    save_csv(
        f"../../results/data/fibers_mean_error_{i}.csv",
        x=torch.arange(1, error.shape[0] + 1),
        y=torch.mean(error, dim=(1, 2, 3)),
    )
