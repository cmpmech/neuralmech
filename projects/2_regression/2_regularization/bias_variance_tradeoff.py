import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ML import PolynomialRegression
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(1)

# select case
CASE = 0
# CASE = 1
# CASE = 2

# ------------------------------- settings -------------------------------
if CASE == 0:  # number of data points scaling
    REGULARIZATION, P = 0, 5
    SAMPLES = 8
    # SAMPLES = 12
    # SAMPLES = 16
elif CASE == 1:  # capacity
    REGULARIZATION, SAMPLES = 0, 16
    P = 2
    # P = 5
    # P = 7
elif CASE == 2:  # regularization
    P, SAMPLES = 7, 16
    REGULARIZATION = 0
    # REGULARIZATION = 1e-5
    # REGULARIZATION = 1e-1

num_fits = 1000
y_true = lambda x: np.cos(np.pi * x)

y_preds = []
for i in range(num_fits):
# --------------------------- data generation ----------------------------
    x_train = rng.uniform(-1, 1, SAMPLES)
    x_train[0] = -1
    x_train[1] = 1
    noise = rng.uniform(-1, 1, SAMPLES) * 0.1
    y_train = y_true(x_train) + noise

# --------------------------------- fit ----------------------------------
    model = PolynomialRegression(P, REGULARIZATION)
    model.fit(x_train, y_train)

# ------------------------------ prediction ------------------------------
    x_pred = np.linspace(-1, 1, 100)
    y_pred = model.forward(x_pred)
    y_preds.append(y_pred)

y_preds = np.vstack(y_preds)
y_pred_mean = np.mean(y_preds, axis=0)
y_pred_std = np.std(y_preds, axis=0)

variance = y_pred_std**2
bias = y_pred_mean - y_true(x_pred)

print(f"bias: {np.mean(np.abs(bias)):.2e}, variance: {np.mean(variance):.2e}")

if not args.book:
# ---------------------------- postprocessing ----------------------------
    # variance
    fig, ax = plt.subplots()
    for i in range(num_fits):
        ax.plot(x_pred, y_preds[i], "k", alpha=0.01)
    ax.plot(x_pred, y_pred_mean, "r")
    ax.fill_between(
        x_pred,  # 95 % confidence interval
        y_pred_mean - 2 * y_pred_std,
        y_pred_mean + 2 * y_pred_std,
        color="r",
        alpha=0.2,
    )
    ax.plot(x_pred, y_pred_mean - 2 * y_pred_std, "r")
    ax.plot(x_pred, y_pred_mean + 2 * y_pred_std, "r")
    ax.set_ylim(-2, 2)
    plt.show()

    # bias
    fig, ax = plt.subplots()
    for i in range(num_fits):
        ax.plot(x_pred, y_preds[i], "k", alpha=0.01)
    ax.plot(x_pred, y_pred_mean, "b")
    ax.plot(x_pred, y_true(x_pred), "k")
    ax.fill_between(
        x_pred,  # bias
        y_true(x_pred),
        y_pred_mean,
        color="b",
        alpha=0.2,
    )

    ax.set_ylim(-2, 2)
    plt.show()
else:
# ------------------------- book postprocessing --------------------------
    data = {"x": x_pred, "mean": y_pred_mean, "std": y_pred_std, "true": y_true(x_pred)}

    for i in range(100):
        data[f"y_pred_{i}"] = y_preds[i]
    if CASE == 0:
        save_csv(CSV_DIR / f"polynomial_regression_{CASE}_{SAMPLES}.csv", **data)
    elif CASE == 1:
        save_csv(CSV_DIR / f"polynomial_regression_{CASE}_{P}.csv", **data)
    elif CASE == 2:
        save_csv(
            CSV_DIR / f"polynomial_regression_{CASE}_{REGULARIZATION}.csv", **data
        )
