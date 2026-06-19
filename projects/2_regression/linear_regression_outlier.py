import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

rng = np.random.default_rng(2)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# select case
# CASE = 0  # mse
# CASE = 1  # mae
CASE = 2  # huber

# ------------------------------------ create data ------------------------------------
x_train = rng.standard_normal(16)
y_train = 2 * x_train + 3 + rng.standard_normal(16)

# outlier
x_train[0] = -2.5
y_train[0] += 10

x_val = rng.standard_normal(4)
y_val = 2 * x_val + 3 + rng.standard_normal(4)

# -------------------------------------- fitting --------------------------------------
if CASE == 0:
    model = LinearRegression()
elif CASE == 1:
    model = QuantileRegressor(quantile=0.5, alpha=0)
elif CASE == 2:
    model = HuberRegressor()
model.fit(x_train.reshape(-1, 1), y_train)
print(f"coefficients {model.coef_[0]}, intercept {model.intercept_}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    x_test = np.linspace(-3, 3, 2)
    y_test_pred = model.predict(x_test.reshape(-1, 1))
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test_pred, "k")
    ax.plot(x_train, y_train, "ko")
    ax.plot(x_val, y_val, "ro")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(RESULTS_DIR / "linear_regression_outlier_train.csv", x=x_train, y=y_train)
    save_csv(RESULTS_DIR / "linear_regression_outlier_val.csv", x=x_val, y=y_val)
