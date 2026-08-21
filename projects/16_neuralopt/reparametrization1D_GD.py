import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


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
        self.w = torch.nn.Parameter(torch.tensor(init))

    def forward(self):
        return self.w


# --------------------------------------- helper --------------------------------------
def optimize(model, lr, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = np.zeros((epochs, 2))
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = model()
        y = f(x)
        history[epoch] = [x.detach(), y.detach()]

        y.backward()
        optimizer.step()
    return history


# -------------------------------------- settings -------------------------------------
# problem
RANGE = [-1.0, 1.0]
GUESS = 1.0

# optimization
EPOCHS = 500

# initialization (results should be consistent if this is changed)
SEED = 1
SEEDS = 5

# postprocessing
RESOLUTION = 200


torch.manual_seed(SEED)

# ------------------------------------- objective -------------------------------------
f = lambda x: (
    (2 - torch.cos(5 * torch.pi * x) * torch.sin(4 * torch.pi * x))
    * torch.sqrt(x**2 + 1e-4)
)

# ------------------------------------ optimization -----------------------------------
# linear (normal), representative of the stable band, which ends at 1.3e-1
lr = 1e-1
model = Linear(GUESS)
history_linear = optimize(model, lr, EPOCHS)

# mlp, the rate follows from the step wanted in x, so it does not depend on the seed
step = 4.7e-2
layers = [10, 256, 1]
gain = 1.5
activations = [nn.GELU() for i in range(len(layers) - 2)] + [None]
model = MLP(layers, activations, GUESS, gain)
history_mlp = optimize(model, step / model.amplification(), EPOCHS)

ys_mlp = np.zeros(SEEDS)
for seed in range(SEEDS):
    model = MLP(layers, activations, GUESS, gain)
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
