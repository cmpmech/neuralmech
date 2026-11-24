import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(2)

# -------------------------------------- settings -------------------------------------
N_TRAIN = 200
N_TEST = 100
NOISE = 0.1
BINS = np.linspace(-0.8, 0.8, 33)


# --------------------------------------- helper --------------------------------------
def create_data(n):
    X = rng.uniform(-1, 1, (n, 2))
    y = X[:, 0] ** 2 + X[:, 1] + NOISE * rng.standard_normal(n)
    return X, y


def quadratic_features(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.stack([x1, x2, x1**2, x1 * x2, x2**2], axis=1)


def fit_least_squares(X, y):
    X = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict(X, w):
    return X @ w[:-1] + w[-1]


# ------------------------------------ create data ------------------------------------
X_train, y_train = create_data(N_TRAIN)
X_test, y_test = create_data(N_TEST)

# -------------------------------------- training -------------------------------------
w_linear = fit_least_squares(X_train, y_train)
w_quadratic = fit_least_squares(quadratic_features(X_train), y_train)

y_linear = predict(X_test, w_linear)
y_quadratic = predict(quadratic_features(X_test), w_quadratic)

# ----------------------------------- postprocessing ----------------------------------
for name, y_pred in [("linear", y_linear), ("quadratic", y_quadratic)]:
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"{name}: rmse {rmse:.3f}, r2 {r2:.3f}")

if not args.book:
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(y_test, y_linear, "bo")
    ax[0].plot(y_test, y_quadratic, "ko")
    ax[0].plot([-1, 2], [-1, 2], "r--")
    ax[1].hist(y_test - y_linear, BINS, color="b", alpha=0.5)
    ax[1].hist(y_test - y_quadratic, BINS, color="k", alpha=0.5)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "regression_parity.csv",
        y=y_test,
        linear=y_linear,
        quadratic=y_quadratic,
    )
    hist_linear = np.histogram(y_test - y_linear, BINS)[0] / N_TEST
    hist_quadratic = np.histogram(y_test - y_quadratic, BINS)[0] / N_TEST
    save_csv(
        CSV_DIR / "regression_parity_hist.csv",
        x=BINS,
        linear=np.append(hist_linear, hist_linear[-1]),
        quadratic=np.append(hist_quadratic, hist_quadratic[-1]),
    )
