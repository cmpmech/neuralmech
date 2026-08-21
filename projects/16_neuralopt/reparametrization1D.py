import matplotlib.pyplot as plt
import numpy as np
import torch

from reparametrization_config import GUESS, RANGE, Linear, MLP, f, optimize

# -------------------------------------- settings -------------------------------------
# architecture
ARCHITECTURE = dict(depth=1, width=10, activation="GELU", input_dim=10)

# optimization
EPOCHS = 50

# initialization (results should be consistent if this is changed)
SEED = 1
SEEDS = 5

# postprocessing
RESOLUTION = 200


torch.manual_seed(SEED)

# ------------------------------------ optimization -----------------------------------
# linear (normal)
lr = 2e-1
model = Linear(GUESS)
history_linear = optimize(model, lr, EPOCHS)

# mlp
lr = 1e-1
model = MLP(GUESS, **ARCHITECTURE)
history_mlp = optimize(model, lr, EPOCHS)

ys_mlp = np.zeros(SEEDS)
for seed in range(SEEDS):
    model = MLP(GUESS, **ARCHITECTURE)
    ys_mlp[seed] = optimize(model, lr, EPOCHS)[-1, 0]


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
