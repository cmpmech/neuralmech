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
USE_NORMALIZATION = True
NORMALIZATION = "layernorm"  # batchnorm
HIDDEN_LAYERS = 40
NEURONS = 100
ACTIVATION = nn.ReLU

LAYERS = [1] + [NEURONS] * HIDDEN_LAYERS + [1]
ACTIVATIONS = [ACTIVATION() for _ in range(len(LAYERS) - 2)]
if NORMALIZATION == "batchnorm":
    NORMALIZATIONS = [nn.BatchNorm1d(LAYERS[i], affine=False) for i in range(1, len(LAYERS) - 1)]
elif NORMALIZATION == "layernorm":
    NORMALIZATIONS = [nn.LayerNorm(LAYERS[i]) for i in range(1, len(LAYERS) - 1)]

# --------------------------- instantiate model & optimizer ---------------------------
# the normalization layer adds a module per block, shifting the skip indices
if USE_NORMALIZATION:
    base_model = MLP(LAYERS, ACTIVATIONS, NORMALIZATIONS)
    skip_connections = [(3 * i + 3, 3 * i + 5) for i in range(HIDDEN_LAYERS - 1)]
else:
    base_model = MLP(LAYERS, ACTIVATIONS)
    skip_connections = [(2 * i + 2, 2 * i + 3) for i in range(HIDDEN_LAYERS - 1)]
model = ResNet(base_model, skip_connections)
model.to(device)
init_weights(model, ACTIVATIONS[0])

# ------------------------------------- prediction ------------------------------------
# hook the normalization (or linear) outputs to read their activation magnitude
activations_avgs = []
hooks = []


def hook_fn(module, input, output):
    activations_avgs.append(torch.mean(torch.abs(output)).item())


if USE_NORMALIZATION:
    module_to_hook = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[NORMALIZATION]
else:
    module_to_hook = nn.Linear
modules = len(model.model)
for i, module in enumerate(model.model):
    if i == modules - 1:
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn))
    elif isinstance(module, module_to_hook):
        hooks.append(module.register_forward_hook(hook_fn))
    if isinstance(module, ResidualBlock):
        for submodule in module.module:
            if isinstance(submodule, module_to_hook):
                hooks.append(submodule.register_forward_hook(hook_fn))

x = torch.randn((100, 1)).to(device)
y_pred = model(x)

for hook in hooks:
    hook.remove()

# ----------------------------------- postprocessing ----------------------------------
layer_ids = np.arange(HIDDEN_LAYERS + 1)
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(layer_ids, activations_avgs, "k")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / f"act_norms_{NORMALIZATION}_{USE_NORMALIZATION}.csv",
        x=layer_ids,
        y=activations_avgs,
    )
