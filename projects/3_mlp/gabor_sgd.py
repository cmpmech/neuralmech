import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

np.random.seed(0)  # used instead of rng, as example was already tuned


# ---------------------------- Gabor helpers -----------------------------
def Gabor(a, b):
    return lambda x: (
        np.sin(a + 0.06 * b * x) * np.exp(-((a + 0.06 * b * x) ** 2) / 32.0)
    )


def cost(ypred, y):
    return np.mean((y - ypred) ** 2)


def cost_grad(a, b, x, y):
    ypred = Gabor(a, b)(x)
    dc_dypred = y - ypred
    sin = np.sin(a + 0.06 * b * x)
    cos = np.cos(a + 0.06 * b * x)
    gaussian = np.exp(-((a + 0.06 * b * x) ** 2) / 32)
    dypred_da = -sin * gaussian * (a + 0.06 * b * x) / 16.0 + cos * gaussian
    dypred_db = (
        -sin * gaussian * 0.06 * x * (a + 0.06 * b * x) / 16.0
        + 0.06 * x * cos * gaussian
    )
    return np.array(
        [-2 * np.mean(dc_dypred * dypred_da), -2 * np.mean(dc_dypred * dypred_db)]
    )


def optimize(params0, lr, epochs, batch_size):
    param_history = []
    param_history.append(params0.copy())
    indices = np.arange(len(x))
    params = params0.copy()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for batch in range(SAMPLES // batch_size):
            batchindices = indices[batch * batch_size : (batch + 1) * batch_size]
            grad = cost_grad(params[0], params[1], x[batchindices], y[batchindices])
            params -= lr * grad
            param_history.append(params.copy())
    param_history = np.vstack(param_history)
    return param_history


# ----------------------------- fitting data -----------------------------
SAMPLES = 64
NOISE = 0.0
x = np.random.uniform(-10, 10, SAMPLES)
y = Gabor(0, 16)(x) + np.random.uniform(-NOISE, NOISE, SAMPLES)


# -------------------- sample optimization landscape ---------------------
samples_landscape = 128
a = np.linspace(-10, 10, samples_landscape)
b = np.linspace(1e-4, 20, samples_landscape)
a, b = np.meshgrid(a, b, indexing="ij")

cost_landscape = np.zeros_like(a)
for i in range(len(a)):
    for j in range(len(a[0])):
        y_pred = Gabor(a[i, j], b[i, j])(x)
        cost_landscape[i, j] = cost(y_pred, y)


# ---------------------- optimization trajectories -----------------------
LR = 1e0
EPOCHS = 500

# select batch_size
# BATCH_SIZE = 1
# BATCH_SIZE = 16
BATCH_SIZE = 64

params0 = np.array([-1.5, 2.0])
history0 = optimize(params0, LR, EPOCHS, BATCH_SIZE)
params1 = np.array([3.0, 18.0])
history1 = optimize(params1, LR, EPOCHS, BATCH_SIZE)
params2 = np.array([-3.1, 10.0])
history2 = optimize(params2, LR, EPOCHS, BATCH_SIZE)
params3 = np.array([2.7, 12.0])
history3 = optimize(params3, LR, EPOCHS, BATCH_SIZE)


# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
cb = ax.contourf(a, b, cost_landscape, levels=36, cmap="cividis")
ax.plot(history0[:, 0], history0[:, 1], "brown", linewidth=2, alpha=0.8)
ax.plot(history0[0, 0], history0[0, 1], "o", color="brown", ms=8)
ax.plot(history2[:, 0], history2[:, 1], "salmon", linewidth=2, alpha=0.8)
ax.plot(history2[0, 0], history2[0, 1], "o", color="salmon", ms=8)
ax.plot(history3[:, 0], history3[:, 1], "red", linewidth=2, alpha=0.8)
ax.plot(history3[0, 0], history3[0, 1], "o", color="red", ms=8)
ax.plot(history1[:, 0], history1[:, 1], "maroon", linewidth=2, alpha=0.8)
ax.plot(history1[0, 0], history1[0, 1], "o", color="maroon", ms=8)
ax.plot(0, 16, "ws", ms=8)
ax.set_xlim(-10, 10)
ax.set_ylim(0, 20)
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if not args.book:
    plt.show()
else:
# ------------------------- book postprocessing --------------------------
    plt.savefig(RESULTS_DIR / f"gabor_landscape_{BATCH_SIZE}.pdf")
    plt.close()

    save_csv(RESULTS_DIR / f"gabor_data.csv", x=x.squeeze(), y=y.squeeze())

    if BATCH_SIZE == 16:
        x_ = np.linspace(-30, 30, 400)
        for i, (a, b) in enumerate(history0[:500:40]):
            y_ = Gabor(a, b)(x_)
            save_csv(RESULTS_DIR / f"gabor_prediction_{i}.csv", x=x_, y=y_)
