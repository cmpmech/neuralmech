import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ML import LogisticRegression
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

rng = np.random.default_rng(2)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
LR = 1e0
EPOCHS = 200

# ------------------------------------ create data ------------------------------------
x_train = rng.standard_normal((40, 2))
y_train = (
    2 * x_train[:, 0] + x_train[:, 1] > 1 + 0.1 * rng.standard_normal(40)
).astype(np.float64)

x_val = rng.standard_normal((20, 2))
y_val = (2 * x_val[:, 0] + x_val[:, 1] > 1 + 0.1 * rng.standard_normal(20)).astype(
    np.float64
)

# -------------------------------------- training -------------------------------------
model = LogisticRegression()
train_cost, val_cost = model.train(EPOCHS, LR, x_train, y_train, x_val, y_val)
print(f"weights={model.weights}, bias={model.bias}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    ax.plot(val_cost, "r")
    plt.show()

    # decision boundary at y=0.5
    x1 = np.linspace(-3, 3, 2)
    x2 = -(model.weights[0] * x1 + model.bias) / model.weights[1]

    fig, ax = plt.subplots()
    ax.plot(x_train[y_train == 0, 0], x_train[y_train == 0, 1], "bs")
    ax.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], "bo")
    ax.plot(x_val[y_val == 0, 0], x_val[y_val == 0, 1], "rs")
    ax.plot(x_val[y_val == 1, 0], x_val[y_val == 1, 1], "ro")
    ax.plot(x1, x2, "k")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "logistic_regression_train0.csv",
        x1=x_train[y_train == 0, 0],
        x2=x_train[y_train == 0, 1],
    )
    save_csv(
        RESULTS_DIR / "logistic_regression_train1.csv",
        x1=x_train[y_train == 1, 0],
        x2=x_train[y_train == 1, 1],
    )
