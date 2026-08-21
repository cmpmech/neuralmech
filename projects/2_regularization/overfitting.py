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

rng = np.random.default_rng(2)

# select case
# CASE = 0
CASE = 1
# CASE = 2

# ------------------------------- settings -------------------------------
if CASE == 0:  # underfitting
    regularization, samples = 0, 8
    p = 1
elif CASE == 1:  # ideal fitting
    regularization, samples = 0, 8
    p = 2
elif CASE == 2:  # overfitting
    regularization, samples = 0, 8
    p = 6

y_true = lambda x: 2 * x**2

# --------------------------- data generation ----------------------------
x_train = rng.uniform(-1, 1, samples)
noise = rng.uniform(-1, 1, samples) * 0.2
y_train = y_true(x_train) + noise

# --------------------------------- fit ----------------------------------
model = PolynomialRegression(p, regularization)
model.fit(x_train, y_train)

# ------------------------------ prediction ------------------------------
x_pred = np.linspace(-1, 1, 100)
y_pred = model.forward(x_pred)

if not args.book:
# ---------------------------- postprocessing ----------------------------
    fig, ax = plt.subplots()
    ax.plot(x_pred, y_pred, "k")
    ax.plot(x_train, y_train, "ro")
    ax.plot(x_pred, y_true(x_pred), "b")
    ax.set_ylim(-0.5, 2.5)
    plt.show()
else:
# ----------------------- postprocessing for book ------------------------
    save_csv(
        CSV_DIR / f"polynomial_overfitting_{CASE}.csv",
        x=x_pred,
        ypred=y_pred,
        y=y_true(x_pred),
    )
    save_csv(
        CSV_DIR / f"polynomial_overfitting_train_{CASE}.csv", x=x_train, y=y_train
    )
