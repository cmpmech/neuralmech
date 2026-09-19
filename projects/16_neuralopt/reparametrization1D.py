import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# problem
RANGE = [-1.0, 1.0]
GUESS = 1.0

# optimization
OPTIMIZER = torch.optim.SGD
# OPTIMIZER = torch.optim.Adam  # use EPOCHS = 50
EPOCHS = 500
LR = 1e-1  # 2e-1 with adam

# model settings
LAYERS = [10, 256, 1]
ACTIVATIONS = [nn.GELU() for _ in range(len(LAYERS) - 2)] + [None]
GAIN = 1.5

# initialization (results should be consistent if this is changed)
SEEDS = 5

# postprocessing
RESOLUTION = 200


# ------------------------------------- objective -------------------------------------
f = lambda x: (
    (2 - torch.cos(5 * torch.pi * x) * torch.sin(4 * torch.pi * x))
    * torch.sqrt(x**2 + 1e-4)
)


# --------------------------------------- models --------------------------------------
class MLP(nn.Module):
    def __init__(self, layers, activations, init, gain):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if activations and activations[i] is not None:
                modules.append(activations[i])
        self.model = nn.Sequential(*modules)
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0)

        self.correction = 0
        self.input = torch.randn(layers[0]).unsqueeze(0)
        with torch.no_grad():
            self.correction = init - self.forward()  # start at initial guess

    def forward(self):
        return self.model(self.input).squeeze() + self.correction

    def amplification(self):
        # gradient descent displaces x by lr * |dx/dtheta|^2
        self.zero_grad()
        self.forward().backward()
        norm = sum((p.grad**2).sum() for p in self.parameters())
        self.zero_grad()
        return norm.item()


class Linear(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init))

    def forward(self):
        return self.w


# --------------------------------------- helper --------------------------------------
def optimize(model, lr, epochs):
    optimizer = OPTIMIZER(model.parameters(), lr=lr)
    history = np.zeros((epochs, 2))
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = model()
        y = f(x)
        history[epoch] = [x.detach(), y.detach()]

        y.backward()
        optimizer.step()
    return history


# ------------------------------------ optimization -----------------------------------
model = Linear(GUESS)
history_linear = optimize(model, LR, EPOCHS)

step = 4.7e-2
model = MLP(LAYERS, ACTIVATIONS, GUESS, GAIN)
history_mlp = optimize(model, step / model.amplification(), EPOCHS)

ys_mlp = np.zeros(SEEDS)
for seed in range(SEEDS):
    model = MLP(LAYERS, ACTIVATIONS, GUESS, GAIN)
    ys_mlp[seed] = optimize(model, step / model.amplification(), EPOCHS)[-1, 0]


# ----------------------------------- postprocessing ----------------------------------
x_ = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)
y_ = f(x_)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x_, y_, "k")
ax[0].plot(history_linear[:, 0], history_linear[:, 1], "ro")
ax[0].plot(history_mlp[:, 0], history_mlp[:, 1], "b.")

ax[1].hist(ys_mlp, bins=np.linspace(RANGE[0], RANGE[1], 41), color="b")
ax[1].axvline(history_linear[-1, 0], color="r")
ax[1].set_xlim(RANGE[0], RANGE[1])

plt.show()
