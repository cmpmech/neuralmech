import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from NN import MLP as MLPtorch

torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(0)


# ---------------------------------- MLP from scratch ---------------------------------
class MLP:
    def __init__(self, layers, activation, grad_activation):
        self.L = len(layers) - 1

        self.weights = [
            rng.standard_normal((layers[i], layers[i + 1])) for i in range(self.L)
        ]
        self.biases = [rng.standard_normal(layers[i + 1]) for i in range(self.L)]
        self.grad_weights = [None] * self.L
        self.grad_biases = [None] * self.L
        self.act = activation
        self.grad_act = grad_activation
        self.preacts = [None] * (self.L + 1)

# -------------------------------- forward propagation --------------------------------
    def forward(self, x):
        self.preacts[0] = x  # stored for backward
        for l in range(self.L):
            x = x @ self.weights[l] + self.biases[l]
            self.preacts[l + 1] = x  # stored for backward
            x = self.act(x)
        return x

# ---------------------------------- backpropagation ----------------------------------
    def backward(self, grad_cost):
        samples = len(grad_cost)
        deltal = grad_cost * self.grad_act(self.preacts[-1])
        if self.L == 1:  # input layer has no activation
            self.grad_weights[-1] = self.preacts[-2].T @ deltal / samples
        else:
            self.grad_weights[-1] = self.act(self.preacts[-2]).T @ deltal / samples
        self.grad_biases[-1] = np.mean(deltal, 0)

        for l in range(self.L - 2, -1, -1):
            deltal = deltal @ self.weights[l + 1].T * self.grad_act(self.preacts[l + 1])
            if l == 0:  # input layer has no activation
                self.grad_weights[l] = self.preacts[0].T @ deltal / samples
            else:
                self.grad_weights[l] = self.act(self.preacts[l]).T @ deltal / samples
            self.grad_biases[l] = np.mean(deltal, 0)

# ------------------------------ steepest descent update ------------------------------
    def update(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.grad_weights[i]
            self.biases[i] -= lr * self.grad_biases[i]


# -------------------------------------- settings -------------------------------------
# hyperparameters
LR = 5e-2
EPOCHS = 200

# model settings
LAYERS = [1, 24, 24, 24, 1]
activation = np.tanh
grad_activation = lambda x: 1 - np.tanh(x) ** 2

# define loss
cost_fun = lambda y_pred, y: np.mean((y - y_pred) ** 2)
grad_cost_fun = lambda y_pred, y: -2 * (y - y_pred)

# ----------------------------------- training data -----------------------------------
x = np.expand_dims(np.linspace(0, 1, 12), 1)
y = x**2

# --------------------------------- instantiate model ---------------------------------
model = MLP(LAYERS, activation, grad_activation)

# torch model initialized with the same weights, to validate the from-scratch gradients
model_torch = MLPtorch(LAYERS, [nn.Tanh() for _ in range(len(LAYERS) - 1)])
for i, p in enumerate(model_torch.parameters()):
    if i % 2 == 0:
        p.data = torch.from_numpy(model.weights[i // 2].T).clone()
    else:
        p.data = torch.from_numpy(model.biases[i // 2]).clone()

x_torch = torch.from_numpy(x)
y_torch = torch.from_numpy(y)
cost_fun_torch = lambda y_pred, y: torch.mean((y - y_pred) ** 2)

# -------------------------------------- training -------------------------------------
print("from scratch")
cost_history = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    y_pred = model.forward(x)
    cost = cost_fun(y_pred, y)
    cost_history[epoch] = cost
    model.backward(grad_cost_fun(y_pred, y))
    model.update(LR)

    if epoch % 10 == 0:
        print(f"cost = {cost:.2e}")

# ----------------------------------- training torch ----------------------------------
print("torch")
optimizer = torch.optim.SGD(model_torch.parameters(), lr=LR)
cost_history_torch = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_pred_torch = model_torch.forward(x_torch)
    cost = cost_fun_torch(y_pred_torch, y_torch)
    cost_history_torch[epoch] = cost.detach()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"cost = {cost:.2e}")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(dpi=150)
ax.plot(x, y, "k")
ax.plot(x, y_pred, "r--")
ax.plot(x, y_pred_torch.detach(), "r:")
plt.show()

fig, ax = plt.subplots(dpi=150)
ax.set_yscale("log")
ax.plot(cost_history, "k")
ax.plot(cost_history_torch, "r:")
plt.show()
