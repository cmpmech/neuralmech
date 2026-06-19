import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from DL import init_weights
from NN import MLP, ResNet, ResidualBlock
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# model settings
USE_SKIP, USE_INIT = True, True
# USE_SKIP, USE_INIT = True, False
# USE_SKIP, USE_INIT = False, True
# USE_SKIP, USE_INIT = False, False

HIDDEN_LAYERS = 40
NEURONS = 100
ACTIVATION = nn.Sigmoid  # nn.ReLU

LAYERS = [1] + [NEURONS] * HIDDEN_LAYERS + [1]
ACTIVATIONS = [ACTIVATION() for _ in range(len(LAYERS) - 2)]

# --------------------------- instantiate model & optimizer ---------------------------
base_model = MLP(LAYERS, ACTIVATIONS)
if not USE_SKIP:
    model = base_model
else:
    skip_connections = [(2 * i + 2, 2 * i + 3) for i in range(HIDDEN_LAYERS - 1)]
    model = ResNet(base_model, skip_connections)
model.to(device)
if USE_INIT:
    init_weights(model, ACTIVATIONS[0])

# ------------------------------------- prediction ------------------------------------
x = torch.tensor([[1.0]]).to(device)
y_pred = model(x)
y_pred.backward()

# ------------------------------------- gradients -------------------------------------
# mean absolute weight gradient per linear layer, tracing skip blocks in order
avg_gradients = []
for module in model.model:
    if isinstance(module, nn.Linear):
        avg_gradients.append(torch.mean(torch.abs(module.weight.grad)).item())
    if isinstance(module, ResidualBlock):
        for submodule in module.module:
            if isinstance(submodule, nn.Linear):
                avg_gradients.append(torch.mean(torch.abs(submodule.weight.grad)).item())

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(avg_gradients, "k")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    act2string = {nn.ReLU: "relu", nn.Sigmoid: "sigmoid"}
    save_csv(
        RESULTS_DIR / f"avg_gradients_{act2string[ACTIVATION]}_{USE_SKIP}_{USE_INIT}.csv",
        x=np.arange(0, HIDDEN_LAYERS + 1),
        y=avg_gradients,
    )
