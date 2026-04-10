from NN import MLP, ResNet, ResidualBlock
from DL import init_weights
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from postprocessing import save_csv

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

# ------------------------------- NN model -------------------------------
use_normalization = True
# normalization = 'batchnorm'
normalization = 'layernorm'
num_layers = 40
neurons = 100
activation = nn.ReLU

layers = [1] + [neurons] * num_layers + [1]
activations = [activation() for _ in range(len(layers) - 2)]
if normalization == 'batchnorm':
    normalizations = [nn.BatchNorm1d(layers[i], affine=False) for i in range(1, len(layers) - 1)]
elif normalization == 'layernorm':
    normalizations = [nn.LayerNorm(layers[i]) for i in range(1, len(layers) - 1)]

if use_normalization:
    base_model = MLP(layers, activations, normalizations)
    skip_connections=[(3 * i + 3, 3 * i + 5) for i in range(num_layers - 1)]
else:
    base_model = MLP(layers, activations)
    skip_connections=[(2 * i + 2, 2 * i + 3) for i in range(num_layers - 1)]
model = ResNet(base_model, skip_connections)

init_weights(model, activations[0])

# ------------------------------ prediction ------------------------------
activations_avgs = []
hooks = []
def hook_fn(name):
    def hook(module, input, output):
        activations_avgs.append(torch.mean(torch.abs(output)).item())
    return hook

# Register hooks on Linear layers
if use_normalization:
    module_to_hook = {'batchnorm' : nn.BatchNorm1d, 'layernorm' : nn.LayerNorm}[normalization]
else:
    module_to_hook = nn.Linear
modules = len(model.model)
for i, module in enumerate(model.model):
    if i == modules - 1:
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(f"layer_{i}")))
    elif isinstance(module, module_to_hook):
        hooks.append(module.register_forward_hook(hook_fn(f"layer_{i}")))
    if isinstance(module, ResidualBlock):
        for j, submodule in enumerate(module.module):
            if isinstance(submodule, module_to_hook):
                hooks.append(submodule.register_forward_hook(hook_fn(f"layer_{i}_{j}")))

x = torch.randn((100, 1))
y_pred = model(x)

for hook in hooks:
    hook.remove()

# ---------------------------- postprocessing ----------------------------
layers = np.arange(num_layers + 1)
fig, ax = plt.subplots()
ax.plot(layers, activations_avgs, 'k')
ax.set_yscale('log')
plt.show()

for i, item in enumerate(activations_avgs):
    print(f"{i+1}. {item:.2e}")

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/act_norms_{normalization}_{use_normalization}',
         x=layers, y=activations_avgs)