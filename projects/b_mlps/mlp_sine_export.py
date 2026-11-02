from pathlib import Path

import numpy as np
import torch
from torch import nn

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000
LR = 1e-2
SAMPLES = 64
NOISE = 0.1

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, 24, 24, 24, 1]

# ------------------------------------ create data ------------------------------------
X = 2 * torch.rand(SAMPLES, 1) - 1
Y = torch.sin(2 * torch.pi * X) + NOISE * torch.randn(SAMPLES, 1)

# --------------------------- instantiate model & optimizer ---------------------------
modules = []
for fan_in, fan_out in zip(LAYERS[:-2], LAYERS[1:-1]):
    modules += [nn.Linear(fan_in, fan_out), nn.Tanh()]
modules.append(nn.Linear(LAYERS[-2], LAYERS[-1]))
model = nn.Sequential(*modules).to(device)
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    cost = cost_fun(model(X.to(device)), Y.to(device))
    cost.backward()
    optimizer.step()
print(f"cost {cost.item():.2e}")

# --------------------------------------- export --------------------------------------
with open(MODEL_DIR / "mlp_sine_weights.txt", "w") as file:
    for param in model.parameters():
        np.savetxt(file, param.detach().cpu().numpy().flatten(order="F"))
x = torch.tensor([[0.25]], device=device)
print(f"prediction {model(x).item():.6f}")
