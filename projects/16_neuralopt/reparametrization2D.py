import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# problem
# BENCHMARK = "rosenbrock"
# BENCHMARK = "rastrigin"
BENCHMARK = "ackley"
# BENCHMARK = "levy"
GUESS = [3.0, 3.0]

# optimization
# OPTIMIZER = torch.optim.SGD
OPTIMIZER = torch.optim.Adam
EPOCHS = 300

# tuned per optimizer, benchmark and ansatz, the gradient scales differ by orders
LR_LINEAR, LR_MLP = {
    torch.optim.SGD: {
        "rosenbrock": (2.5e-4, 1.1e-5),
        "rastrigin": (3.0e-1, 1.2e-3),
        "ackley": (1.4e-1, 2.8e-2),
        "levy": (2.8e-2, 2.6e-3),
    },
    torch.optim.Adam: {
        "rosenbrock": (1.6e0, 5.8e-2),
        "rastrigin": (1.6e0, 1.6e0),
        "ackley": (4.3e-1, 1.5e-2),
        "levy": (8.3e-1, 3.0e-2),
    },
}[OPTIMIZER][BENCHMARK]

# model settings
LAYERS = [10, 50, 2]
ACTIVATIONS = [nn.GELU() for _ in range(len(LAYERS) - 2)] + [None]

# initialization (results should be consistent if this is changed)
SEEDS = 20

# postprocessing
RESOLUTION = 300


# ------------------------------------- benchmarks ------------------------------------
BENCHMARKS = {
    "rosenbrock": lambda y: (
        100 * (y[..., 1] - y[..., 0] ** 2) ** 2 + (1 - y[..., 0]) ** 2
    ),
    "rastrigin": lambda y: 20 + (y**2 - 10 * torch.cos(2 * torch.pi * y)).sum(-1),
    "ackley": lambda y: (
        -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (y**2).sum(-1)))
        - torch.exp(0.5 * torch.cos(2 * torch.pi * y).sum(-1))
        + np.e
        + 20
    ),
    "levy": lambda y: (
        torch.sin(3 * torch.pi * (y[..., 0] + 1)) ** 2
        + y[..., 0] ** 2 * (1 + torch.sin(3 * torch.pi * (y[..., 1] + 1)) ** 2)
        + y[..., 1] ** 2 * (1 + torch.sin(2 * torch.pi * (y[..., 1] + 1)) ** 2)
    ),
}
f = BENCHMARKS[BENCHMARK]

OPTIMUM = [1.0, 1.0] if BENCHMARK == "rosenbrock" else [0.0, 0.0]
RANGE_X = [-2.0, 4.0] if BENCHMARK == "rosenbrock" else [-3.5, 3.5]
RANGE_Y = [-1.0, 5.0] if BENCHMARK == "rosenbrock" else [-3.5, 3.5]


# --------------------------------------- models --------------------------------------
class MLP(nn.Module):
    def __init__(self, layers, activations, init):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if activations and activations[i] is not None:
                modules.append(activations[i])
        self.model = nn.Sequential(*modules)
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
                nn.init.constant_(module.bias, 0)

        self.correction = 0
        self.input = torch.rand(layers[0]).unsqueeze(0) * 2 - 1
        guess = torch.tensor(init)
        with torch.no_grad():
            self.correction = guess - self.forward()  # start at initial guess

    def forward(self):
        return self.model(self.input).squeeze(0) + self.correction


class Linear(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init))

    def forward(self):
        return self.w


# --------------------------------------- helper --------------------------------------
def optimize(model, lr, epochs):
    optimizer = OPTIMIZER(model.parameters(), lr=lr)
    history = np.zeros((epochs, 3))  # x, y, f(x, y)
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = model()
        y = f(x)
        history[epoch] = [*x.detach(), y.detach()]

        y.backward()
        optimizer.step()
    return history


# ------------------------------------ optimization -----------------------------------
# linear (normal)
model = Linear(GUESS)
history_linear = optimize(model, LR_LINEAR, EPOCHS)

# mlp, rerun over SEEDS initializations; the first carries the plotted path
costs_mlp = np.zeros(SEEDS)
for seed in range(SEEDS):
    model = MLP(LAYERS, ACTIVATIONS, GUESS)
    history = optimize(model, LR_MLP, EPOCHS)
    costs_mlp[seed] = history[-1, 2]
    if seed == 0:
        history_mlp = history

# a fixed rate can still diverge on some initializations
finite = np.isfinite(costs_mlp)
cost_linear = history_linear[-1, 2]

print(f"{BENCHMARK}, {OPTIMIZER.__name__}, {EPOCHS} epochs")
print(f"  linear      f = {cost_linear:.2e}")
print(f"  mlp median  f = {np.median(costs_mlp[finite]):.2e} over {finite.sum()} seeds")
print(f"  mlp beat linear in {int((costs_mlp < cost_linear).sum())}/{SEEDS}")
print(f"  mlp diverged in {SEEDS - int(finite.sum())}/{SEEDS}")

# ----------------------------------- postprocessing ----------------------------------
x_ = torch.linspace(RANGE_X[0], RANGE_X[1], RESOLUTION)
y_ = torch.linspace(RANGE_Y[0], RANGE_Y[1], RESOLUTION)
x_, y_ = torch.meshgrid(x_, y_, indexing="ij")
z_ = f(torch.stack([x_, y_], dim=-1))

norm = colors.LogNorm() if BENCHMARK == "rosenbrock" else colors.Normalize()

fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(x_, y_, z_, norm=norm, cmap="cividis", shading="auto", rasterized=True)
ax[0].contour(x_, y_, z_, norm=norm, colors="k", linewidths=0.4)
ax[0].plot(OPTIMUM[0], OPTIMUM[1], "ws", markersize=8)
ax[0].plot(history_linear[:, 0], history_linear[:, 1], "r", linewidth=2)
ax[0].plot(history_linear[-1, 0], history_linear[-1, 1], "ro")
ax[0].plot(history_mlp[:, 0], history_mlp[:, 1], "b", linewidth=2)
ax[0].plot(history_mlp[-1, 0], history_mlp[-1, 1], "bo")
ax[0].set_xlim(RANGE_X[0], RANGE_X[1])
ax[0].set_ylim(RANGE_Y[0], RANGE_Y[1])
ax[0].set_aspect("equal", adjustable="box")

# the costs span decades, so bin them logarithmically
floor = 1e-12
edges = np.concatenate([costs_mlp[finite], [cost_linear]]).clip(floor)
bins = np.logspace(np.log10(edges.min()), np.log10(edges.max()), 41)
ax[1].hist(costs_mlp[finite].clip(floor), bins=bins, color="b")
ax[1].axvline(cost_linear.clip(floor), color="r")
ax[1].set_xscale("log")

plt.show()
