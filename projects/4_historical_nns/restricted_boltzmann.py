import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from helper import bars_and_stripes, contrastive_divergence

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 10000
LR = 0.05

# model settings
RESOLUTION = 4
HIDDEN = 32
GIBBS_STEPS = 1000
SAMPLES = 16

# ------------------------------------ create data ------------------------------------
X = bars_and_stripes(RESOLUTION).to(device)
D = X.shape[1]

# --------------------------- instantiate model & optimizer ---------------------------
W = 0.01 * torch.randn(D, HIDDEN, device=device)
b = torch.zeros(D, device=device)
c = torch.zeros(HIDDEN, device=device)

# -------------------------------------- training -------------------------------------
for epoch in tqdm(range(EPOCHS)):
    contrastive_divergence(X, W, b, c, LR)

# ----------------------------------- postprocessing ----------------------------------
v = torch.bernoulli(0.5 * torch.ones(SAMPLES, D, device=device))
for _ in range(GIBBS_STEPS):
    h = torch.bernoulli(torch.sigmoid(v @ W + c))
    v = torch.bernoulli(torch.sigmoid(h @ W.T + b))

valid = (v[:, None, :] == X[None, :, :]).all(dim=2).any(dim=1)
print(f"valid bars and stripes among samples {int(valid.sum())}/{SAMPLES}")

fig, axes = plt.subplots(2, SAMPLES // 2, figsize=(SAMPLES // 2, 2))
for ax, sample in zip(axes.flatten(), v):
    ax.imshow(sample.view(RESOLUTION, RESOLUTION).cpu(), cmap="binary")
    ax.axis("off")
plt.show()
