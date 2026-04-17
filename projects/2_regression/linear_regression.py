import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ML import LinearRegression
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(2)

# --------------------------- data generation ----------------------------
x_train = rng.standard_normal(16)
y_train = 2 * x_train + 3 + rng.standard_normal(16)

x_val = rng.standard_normal(4)
y_val = 2 * x_val + 3 + rng.standard_normal(4)

# ----------------------------- optimization -----------------------------
model = LinearRegression()
LR = 1e-2
EPOCHS = 200

train_cost, val_cost = model.train(EPOCHS, LR, x_train, y_train, x_val, y_val)

print(f"weight w={model.weight} & bias b={model.bias}")

# --------------------------- normal equations ---------------------------
X = np.vstack((x_train, np.ones_like(x_train))).T
y = y_train

weight, bias = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"weight w={weight} & bias b={bias}")

if not args.book:
# ---------------------------- postprocessing ----------------------------
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    plt.show()

    x_test = np.linspace(-3, 3, 2)
    y_test_pred = model.forward(x_test)
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test_pred, "k")
    ax.plot(x_train, y_train, "ko")
    ax.plot(x_val, y_val, "ro")
    plt.show()
else:
# ------------------------- book postprocessing --------------------------
    save_csv(RESULTS_DIR / "linear_regression_train.csv", x=x_train, y=y_train)
    save_csv(RESULTS_DIR / "linear_regression_val.csv", x=x_val, y=y_val)
