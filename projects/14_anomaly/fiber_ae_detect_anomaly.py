from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# ------------------------------ load data -------------------------------
domain_size = 256

data = []
for i in range(10):
    data.append(torch.from_numpy(
        np.load(BASE_DIR / f"../../data/fibers_anomaly_{i}_{domain_size}.npy")))
    data[-1] = data[-1].to(torch.float32).unsqueeze(1).to(device)

# -------------------------- load trained model --------------------------
bottleneck_layers = 1 #1

model = torch.load(BASE_DIR / f'../../models/fiber_ae_{bottleneck_layers}_{domain_size}.pt2', weights_only=False, map_location=device)
model.eval()
standardizex = model.standardizer

# --------------------------- reconstructions ----------------------------
reconstructions = []
errors = []
with torch.no_grad():
    for fibers in data:
        reconstructions.append(standardizex.inverse(model(standardizex(fibers))).cpu())
        errors.append((reconstructions[-1] - fibers.cpu())**2)

# ---------------------------- postprocessing ----------------------------
anomaly_deg = 1 # 0 does not work
sample = 0

fig, ax = plt.subplots()
ax.imshow(data[anomaly_deg][sample][0].T.cpu(), origin='lower',
          cmap='binary', vmin=0, vmax=1)
plt.show()

fig, ax = plt.subplots()
ax.imshow(reconstructions[anomaly_deg][sample][0].T, origin='lower',
          cmap='binary', vmin=0, vmax=1)
plt.show()

fig, ax = plt.subplots()
ax.imshow(errors[anomaly_deg][sample][0].T, origin='lower',
          cmap='hot_r', vmin=0)
plt.show()

# TODO INCREASE MINIMUM FIBER SIZE
