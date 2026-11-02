import jax
import jax.numpy as jnp
import numpy as np
import optax

key = jax.random.key(0)

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000
LR = 1e-2
SAMPLES = 64
NOISE = 0.1

# model settings
LAYERS = [1, 24, 24, 24, 1]

# ------------------------------------ create data ------------------------------------
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (SAMPLES, 1))
Y = np.sin(2 * np.pi * X) + NOISE * rng.standard_normal((SAMPLES, 1))


def init_params(key, layers):
    params = []
    for fan_in, fan_out in zip(layers[:-1], layers[1:]):
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (fan_in, fan_out))
        params.append((W / jnp.sqrt(fan_in), jnp.zeros(fan_out)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def cost_fun(params, x, y):
    return jnp.mean((mlp(params, x) - y) ** 2)


# --------------------------- instantiate model & optimizer ---------------------------
params = init_params(key, LAYERS)
optimizer = optax.adam(LR)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, x, y):
    cost, grads = jax.value_and_grad(cost_fun)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, cost


# -------------------------------------- training -------------------------------------
for epoch in range(EPOCHS):
    params, opt_state, cost = train_step(params, opt_state, X, Y)
print(f"cost {cost:.2e}")
