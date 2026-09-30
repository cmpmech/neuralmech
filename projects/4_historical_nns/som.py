import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 20
LR = 0.5  # decays linearly to zero
SIGMA = 3.0  # neighborhood width in grid units, decays linearly to 0.5

# model settings
RESOLUTION = 10  # RESOLUTION x RESOLUTION grid of neurons
SAMPLES = 1000

# ------------------------------------ create data ------------------------------------
angle = 2 * torch.pi * torch.rand(SAMPLES)
radius = 0.7 + 0.3 * torch.rand(SAMPLES)
X = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1).to(device)

# --------------------------- instantiate model & optimizer ---------------------------
i, j = torch.meshgrid(torch.arange(RESOLUTION), torch.arange(RESOLUTION), indexing="ij")
grid = torch.stack([i.flatten(), j.flatten()], dim=1).float().to(device)
M = 0.1 * torch.randn(RESOLUTION**2, 2, device=device)

# -------------------------------------- training -------------------------------------
steps = EPOCHS * SAMPLES
for step in range(steps):
    progress = step / steps
    lr = LR * (1 - progress)
    sigma = SIGMA + (0.5 - SIGMA) * progress
    x = X[torch.randint(0, SAMPLES, ())]
    winner = ((M - x) ** 2).sum(dim=1).argmin()
    neighborhood = torch.exp(-((grid - grid[winner]) ** 2).sum(dim=1) / (2 * sigma**2))
    M += lr * neighborhood[:, None] * (x - M)

# ----------------------------------- postprocessing ----------------------------------
M = M.view(RESOLUTION, RESOLUTION, 2).cpu()
fig, ax = plt.subplots()
ax.plot(X[:, 0].cpu(), X[:, 1].cpu(), "b.", alpha=0.2)
ax.plot(M[:, :, 0], M[:, :, 1], "k")
ax.plot(M[:, :, 0].T, M[:, :, 1].T, "k")
ax.set_aspect("equal")
plt.show()
