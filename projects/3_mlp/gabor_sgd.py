import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

np.random.seed(0)  # used instead of rng, as example was already tuned

# -------------------------------------- settings -------------------------------------
# hyperparameters
LR = 1e0
EPOCHS = 500

# select batch_size
# BATCH_SIZE = 1
# BATCH_SIZE = 16
BATCH_SIZE = 64

# fitting data
SAMPLES = 64
NOISE = 0.0

# resolution of the sampled loss landscape
RESOLUTION = 128


# --------------------------------------- helper --------------------------------------
def gabor(a, b):
    return lambda x: (
        np.sin(a + 0.06 * b * x) * np.exp(-((a + 0.06 * b * x) ** 2) / 32.0)
    )


def cost(y_pred, y):
    return np.mean((y - y_pred) ** 2)


def cost_grad(a, b, x, y):
    y_pred = gabor(a, b)(x)
    dc_dypred = y - y_pred
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
    params = params0.copy()
    param_history = [params.copy()]
    ids = np.arange(len(x))
    for epoch in range(epochs):
        np.random.shuffle(ids)
        for batch in range(SAMPLES // batch_size):
            batch_ids = ids[batch * batch_size : (batch + 1) * batch_size]
            grad = cost_grad(params[0], params[1], x[batch_ids], y[batch_ids])
            params -= lr * grad
            param_history.append(params.copy())
    return np.vstack(param_history)


# ----------------------------------- create data -------------------------------------
x = np.random.uniform(-10, 10, SAMPLES)
y = gabor(0, 16)(x) + np.random.uniform(-NOISE, NOISE, SAMPLES)

# ------------------------------- sample loss landscape -------------------------------
a = np.linspace(-10, 10, RESOLUTION)
b = np.linspace(1e-4, 20, RESOLUTION)
a, b = np.meshgrid(a, b, indexing="ij")

cost_landscape = np.zeros_like(a)
for i in range(len(a)):
    for j in range(len(a[0])):
        y_pred = gabor(a[i, j], b[i, j])(x)
        cost_landscape[i, j] = cost(y_pred, y)

# ----------------------------- optimization trajectories -----------------------------
history0 = optimize(np.array([-1.5, 2.0]), LR, EPOCHS, BATCH_SIZE)
history1 = optimize(np.array([3.0, 18.0]), LR, EPOCHS, BATCH_SIZE)
history2 = optimize(np.array([-3.1, 10.0]), LR, EPOCHS, BATCH_SIZE)
history3 = optimize(np.array([2.7, 12.0]), LR, EPOCHS, BATCH_SIZE)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
ax.contourf(a, b, cost_landscape, levels=36, cmap="cividis")
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

# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / f"gabor_landscape_{BATCH_SIZE}.pdf")
    plt.close()

    save_csv(CSV_DIR / "gabor_data.csv", x=x.squeeze(), y=y.squeeze())

    if BATCH_SIZE == 16:
        x_test = np.linspace(-30, 30, 400)
        for i, (a, b) in enumerate(history0[:500:40]):
            y_pred = gabor(a, b)(x_test)
            save_csv(CSV_DIR / f"gabor_prediction_{i}.csv", x=x_test, y=y_pred)
