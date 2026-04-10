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
use_skip, use_init = True, True
# use_skip, use_init = True, False
# use_skip, use_init = False, True
# use_skip, use_init = False, False

num_layers = 40
neurons = 100
# activation = nn.ReLU
activation = nn.Sigmoid

layers = [1] + [neurons] * num_layers + [1]
activations = [activation() for _ in range(len(layers) - 2)]

base_model = MLP(layers, activations)
if use_skip == False:
    model = base_model
else:
    skip_connections=[(2 * i + 2, 2 * i + 3) for i in range(num_layers - 1)]
    model = ResNet(base_model, skip_connections)

if use_init == True:
    init_weights(model, activations[0])

# ------------------------------ prediction ------------------------------
x = torch.tensor([[1.0]])
y = torch.tensor([[1.0]])
y_pred = model(x)

y_pred.backward()

# ------------------------------ gradients -------------------------------
avg_gradients = []
for module in model.model:
    if isinstance(module, nn.Linear):
        avg_gradients.append(torch.mean(torch.abs(module.weight.grad)).item())
    if isinstance(module, ResidualBlock):
        for submodule in module.module:
            if isinstance(submodule, nn.Linear):
                avg_gradients.append(torch.mean(torch.abs(submodule.weight.grad)).item())

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(avg_gradients)
ax.set_yscale('log')
plt.show()

for i, item in enumerate(avg_gradients):
    print(f"{i+1}. {item:.2e}")

# ------------------------- book postprocessing --------------------------
maptostring = {nn.ReLU : 'relu', nn.Sigmoid : 'sigmoid'}
save_csv(f'../../results/avg_gradients_{maptostring[activation]}_{use_skip}_{use_init}.csv',
         x=np.arange(0, num_layers + 1), y=avg_gradients)