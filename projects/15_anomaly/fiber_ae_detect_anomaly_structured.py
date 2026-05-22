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

data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/fibers_anomaly_structured_{domain_size}.npy")
)
data = data.to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)

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
# reconstructions = []
# errors = []
with torch.no_grad():
    reconstruction = standardizex.inverse(model(standardizex(data))).cpu()
    error = (reconstruction - data.cpu()) ** 2

# ------------------------- book postprocessing --------------------------
# reconstructions
fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
ax.imshow(data[0, 0].T.cpu(), origin="lower", cmap="binary", vmin=0, vmax=1)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/fibers_detection_true_structured.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
ax.imshow(
    reconstruction[0, 0].T,
    origin="lower",
    cmap="binary",
    vmin=0,
    vmax=1,
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/fibers_detection_pred_structured.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
ax.imshow(
    error[0, 0].T,
    origin="lower",
    cmap="hot_r",
    norm=LogNorm(vmin=1e-5, vmax=1.3),
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/fibers_detection_error_structured.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

mse = torch.mean(error, dim=(1, 2, 3))
print(f"mse: {mse.item():.3e}")

# # -------------------------------- export --------------------------------
# for i, error in enumerate(errors):
#     save_csv(
#         f"../../results/fibers_mean_error_{i}.csv",
#         x=torch.arange(1, error.shape[0] + 1),
#         y=torch.mean(error, dim=(1, 2, 3)),
#     )
