"""Shared pieces of the 1D neural reparametrization study.

Imported by reparametrization1D.py (the comparison figure) and
reparametrization1D_tune.py (the hyperparameter search). Every hyperparameter is a
keyword argument whose default reproduces the original hand-tuned configuration, so the
figure is unchanged while the search can vary all of them.

Depends on torch/numpy only, deliberately: the repo-level NN/DL modules would supply the
linear stack and the initialization table, but they pull escnn, neuralop and
torch_geometric, which costs ~10 s of import time -- far more than this 1D problem takes
to solve.
"""

import numpy as np
import torch
from torch import nn

# -------------------------------------- problem --------------------------------------
RANGE = [-1.0, 1.0]
GUESS = 1.0

f = lambda x: (
    (2 - torch.cos(5 * torch.pi * x) * torch.sin(4 * torch.pi * x))
    * torch.sqrt(x**2 + 1e-4)
)


# ------------------------------------- registries ------------------------------------
# Named rather than passed as instances so a search axis stays a hashable primitive.
ACTIVATIONS = {
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}

# The gain DL.init_weights would pick per activation; everything else is a rectifier.
SATURATING = {"Tanh": "tanh", "Sigmoid": "sigmoid"}


def initialize(model, scheme, activation, gain):
    """Reinitialize every linear layer; "default" leaves PyTorch's own init alone.

    "matched" mirrors DL.init_weights: Xavier scaled by the saturating activation's gain,
    Kaiming otherwise. Kaiming is not offered as a standalone scheme because it takes no
    gain, which would leave the gain axis undefined for one of its own settings.
    """
    if scheme == "default":
        return

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if scheme == "matched":
            if activation in SATURATING:
                gain_ = nn.init.calculate_gain(SATURATING[activation])
                nn.init.xavier_uniform_(module.weight, gain=gain_)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif scheme == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=gain)
        elif scheme == "xavier":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            raise ValueError(f"unknown init scheme {scheme!r}")
        nn.init.zeros_(module.bias)


# --------------------------------------- models --------------------------------------
class MLP(nn.Module):
    """The design variable reparametrized by a network, x = MLP(constant input).

    The input never changes, so the network is a parametrization rather than a regressor:
    nothing is learned and nothing generalizes. All it does is reshape the landscape the
    optimizer walks, which is what lets it leave a local minimum that plain descent on x
    cannot.

    output_activation defaults to True to match the original script, whose activation
    list ran over every linear layer including the last. That squashes the reachable
    range of x (GELU bottoms out at ~-0.17) and is part of why the reparametrization
    escapes, so it is exposed as a knob rather than quietly corrected.
    """

    def __init__(
        self,
        guess,
        depth=1,
        width=10,
        activation="GELU",
        input_dim=10,
        input_scale=1.0,
        learnable=False,
        scheme="default",
        gain=1.0,
        output_activation=True,
    ):
        super().__init__()
        layers = [input_dim] + [width] * depth + [1]

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if output_activation or i < len(layers) - 2:
                modules.append(ACTIVATIONS[activation]())
        self.model = nn.Sequential(*modules)
        initialize(self.model, scheme, activation, gain)

        # input_scale sets the effective gradient scale, so it acts much like the step size
        x = torch.randn(input_dim).unsqueeze(0) * input_scale
        if learnable:
            self.input = nn.Parameter(x)
        else:
            self.register_buffer("input", x)

        self.correction = 0
        with torch.no_grad():
            self.correction = guess - self.forward()  # start at initial guess

    def forward(self):
        return self.model(self.input).squeeze() + self.correction

    def amplification(self):
        """|dx/dtheta|^2, the factor by which one gradient descent step displaces x."""
        self.zero_grad()
        self.forward().backward()
        norm = sum((p.grad**2).sum() for p in self.parameters() if p.grad is not None)
        self.zero_grad()
        return norm.item()


class Linear(nn.Module):
    """The baseline: optimize the design variable directly."""

    def __init__(self, guess):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(guess))

    def forward(self):
        return self.w


# --------------------------------------- helper --------------------------------------
def optimize(model, lr, epochs, optimizer="Adam"):
    """Descend f(model()) and return the [x, y] history over all epochs."""
    optimizer = OPTIMIZERS[optimizer](model.parameters(), lr=lr)
    history = np.zeros((epochs, 2))
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = model()
        y = f(x)
        history[epoch] = [x.detach(), y.detach()]

        y.backward()
        optimizer.step()
    return history
