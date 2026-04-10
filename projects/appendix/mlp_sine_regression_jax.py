from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import TensorDataset, DataLoader
import torch
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

BASE_DIR = Path(__file__).parent
key = jax.random.PRNGKey(0)
print(jax.devices())

# -------------------------- training settings ---------------------------
epochs = 4000
lr = 1e-2
regularization = 0
batch_size = 32

# ---------------------------- model settings ----------------------------
layers = [1, 24, 24, 24, 1]

# ----------------------------- prepare data -----------------------------
data = np.load(BASE_DIR / '../../data/sine.npz')
X = jnp.array(data['X'], dtype=jnp.float32)
Y = jnp.array(data['Y'], dtype=jnp.float32)
train_loader = DataLoader(TensorDataset(torch.from_numpy(np.array(X)),
                                        torch.from_numpy(np.array(Y))),
                          batch_size=batch_size, shuffle=True)

# ----------------------- MLP (inline, no extra fn) ----------------------
def init_params(key, layer_sizes):
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, k1, k2 = jax.random.split(key, 3)
        W = jax.random.normal(k1, (fan_in, fan_out)) * 0.1
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params

def mlp_forward(params, x):
    for W, b in params[:-1]:
        x = jax.nn.gelu(x @ W + b, approximate=True)
    W, b = params[-1]
    return x @ W + b

# ------------------------------- optimizer ------------------------------
key, subkey = jax.random.split(key)
params = init_params(subkey, layers)
optimizer = optax.adamw(lr, weight_decay=regularization)
opt_state = optimizer.init(params)

# ------------------------------- training -------------------------------
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(
        lambda p: jnp.mean((mlp_forward(p, x) - y) ** 2)
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

train_cost = [0.0] * epochs
tic = time.time()
pbar = tqdm(range(epochs))
for epoch in pbar:
    batch_losses = []
    for x_b, y_b in train_loader:
        params, opt_state, loss = train_step(
            params, opt_state, jnp.array(x_b), jnp.array(y_b))
        batch_losses.append(float(loss))
    train_cost[epoch] = float(np.mean(batch_losses))
    if epoch % 10 == 0:
        pbar.set_postfix({'train': f'{train_cost[epoch]:.2e}'})
print(f'elapsed time {time.time() - tic:.2f} s')

# ---------------------------- postprocessing ----------------------------
plt.figure()
plt.yscale('log')
plt.plot(train_cost, 'k')
plt.show()

x_test = jnp.linspace(-1.3, 1.3, 100)[:, None]
y_test = jnp.sin(2 * jnp.pi * x_test)
y_pred = mlp_forward(params, x_test)

plt.figure()
plt.plot(np.array(x_test), np.array(y_test), 'k')
plt.plot(np.array(X), np.array(Y), 'ko')
plt.plot(np.array(x_test), np.array(y_pred), 'r--')
plt.show()