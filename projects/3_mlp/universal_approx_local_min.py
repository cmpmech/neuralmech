import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()

torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
NEURONS = 20
EPOCHS = 10000
LR = 1e-2
SEEDS = 20

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, NEURONS, 1]
ACTIVATIONS = [nn.ReLU(inplace=True)]

# ----------------------------------- training data -----------------------------------
SAMPLES = NEURONS + 1
x_train = torch.linspace(-1, 1, SAMPLES).unsqueeze(1).to(device)
y_train = torch.sin(torch.pi * x_train).to(device)

# -------------------------------------- training -------------------------------------
# every seed restarts adam from a fresh initialization, ending in a different local min
costs = []
tic = time.time()
pbar = tqdm(range(SEEDS))
for seed in pbar:
    torch.manual_seed(seed)
    model = MLP(LAYERS, post_modules=ACTIVATIONS)
    model.to(device)
    init_weights(model, ACTIVATIONS[0])
    optimizer = torch.optim.Adam(model.parameters(), LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        cost = cost_fun(model(x_train), y_train)
        cost.backward()
        optimizer.step()

    costs.append(cost.item())
    pbar.set_postfix({"train": f"{cost.item():.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
# the least squares construction of universal_approx_relu.py is the global reference
model_ref = torch.load(
    MODEL_DIR / f"universal_approx_relu{NEURONS}.pt2",
    weights_only=False,
    map_location=device,
)
with torch.no_grad():
    cost_ref = cost_fun(model_ref(x_train), y_train).item()

print(f"cost adam worst {max(costs):.2e}")
print(f"cost adam mean {sum(costs) / SEEDS:.2e}")
print(f"cost adam best {min(costs):.2e}")
print(f"cost least squares {cost_ref:.2e}")

fig, ax = plt.subplots()
ax.hist(np.log10(costs), bins=10, color="k")
ax.axvline(np.log10(cost_ref), color="b")
ax.set_xlabel("log10 cost")
plt.show()
