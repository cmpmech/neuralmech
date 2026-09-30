import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
RESOLUTION = 10  # patterns are RESOLUTION x RESOLUTION images, D = RESOLUTION^2 neurons
PATTERNS = 8  # capacity is roughly 0.14 D
CORRUPTION = 0.25  # fraction of flipped pixels
SWEEPS = 10


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
D = RESOLUTION**2
X = (torch.randint(0, 2, (PATTERNS, D)) * 2 - 1).float().to(device)

flip = torch.rand(D, device=device) < CORRUPTION
s_corrupted = torch.where(flip, -X[0], X[0])

# -------------------------------------- training -------------------------------------
W = store(X)

# ----------------------------------- postprocessing ----------------------------------
s_recalled = recall(W, s_corrupted.clone(), SWEEPS)

print(f"energy corrupted {energy(W, s_corrupted):.2e}, recalled {energy(W, s_recalled):.2e}")
print(f"pixel errors corrupted {int(flip.sum())}, recalled {int((s_recalled != X[0]).sum())}")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, s in zip(axes, [X[0], s_corrupted, s_recalled]):
    ax.imshow(s.view(RESOLUTION, RESOLUTION).cpu(), cmap="binary")
    ax.axis("off")
plt.show()
