import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 20  # upper bound, training stops once every sample is classified correctly
LR = 1.0

# -------------------------------------- helper ---------------------------------------
heaviside = lambda z: (z > 0).float()

# ------------------------------------ create data ------------------------------------
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).to(device)
targets = {
    "and": torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device),
    "xor": torch.tensor([0.0, 1.0, 1.0, 0.0]).to(device),
}

# -------------------------------------- training -------------------------------------
params = {}
for name, Y in targets.items():
    w = torch.zeros(2, device=device)
    b = torch.zeros((), device=device)
    for epoch in range(EPOCHS):
        errors = 0
        for x, y in zip(X, Y):
            y_pred = heaviside(x @ w + b)
            w += LR * (y - y_pred) * x
            b += LR * (y - y_pred)
            errors += int(y_pred != y)
        if errors == 0:
            break
    params[name] = (w, b)
    print(f"{name}: {errors} misclassified after {epoch + 1} epochs")

# ----------------------------------- postprocessing ----------------------------------
x1, x2 = torch.meshgrid(
    torch.linspace(-0.5, 1.5, 200), torch.linspace(-0.5, 1.5, 200), indexing="ij"
)
grid = torch.stack([x1.flatten(), x2.flatten()], dim=1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (name, Y) in zip(axes, targets.items()):
    w, b = params[name]
    y_grid = heaviside(grid @ w + b).view(x1.shape)
    ax.contourf(x1, x2, y_grid, levels=[-0.5, 0.5, 1.5], cmap="cividis", alpha=0.4)
    ax.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=Y.cpu(), cmap="cividis", edgecolors="k")
    ax.set_title(name)
    ax.set_aspect("equal")
plt.show()
