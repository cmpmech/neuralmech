from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# model settings
CENTERS = 12
GAMMA = 0.1  # width of the Gaussians

# -------------------------------------- helper ---------------------------------------
features = lambda x, centers: torch.exp(-torch.cdist(x, centers) ** 2 / (2 * GAMMA**2))

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "sine.npz")
X_train = torch.from_numpy(data["X"]).to(torch.float64).to(device)
Y_train = torch.from_numpy(data["Y"]).to(torch.float64).to(device)

# -------------------------------------- training -------------------------------------
kmeans = KMeans(n_clusters=CENTERS, n_init=10, random_state=0).fit(X_train.cpu())
centers = torch.from_numpy(kmeans.cluster_centers_).to(device)

Phi = features(X_train, centers)
Phi = torch.cat([Phi, torch.ones(len(Phi), 1, dtype=Phi.dtype, device=device)], dim=1)
weights = torch.linalg.solve(Phi.T @ Phi, Phi.T @ Y_train)

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-1.3, 1.3, 200, dtype=torch.float64, device=device)[:, None]
y_test = torch.sin(2 * torch.pi * x_test)
Phi_test = features(x_test, centers)
Phi_test = torch.cat([Phi_test, torch.ones(len(x_test), 1, dtype=Phi.dtype)], dim=1)
y_pred_test = Phi_test @ weights

fig, ax = plt.subplots()
ax.plot(x_test.cpu(), y_test.cpu(), "k")
ax.plot(x_test.cpu(), y_pred_test.cpu(), "r--")
ax.plot(X_train.cpu(), Y_train.cpu(), "bo")
ax.plot(centers.cpu(), torch.zeros(CENTERS), "kx")
ax.set_ylim(-2, 2)
plt.show()
