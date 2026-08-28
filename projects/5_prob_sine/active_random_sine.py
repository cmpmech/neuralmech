import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ML import BayesianPolynomialRegression
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
SAMPLES = 2  # initial design size
ACQUISITIONS = 48  # Latin hypercube refinements
REPEATS = 50
RESOLUTION = 1000
NOISE = 0.05

# postprocessing
FRAMES = [0, 2, 4, 6, 8]

# model settings
DEGREE = 9
REGULARIZATION = 1e-2

# function to be learned
f = lambda x: np.sin(2 * np.pi * x)

# ------------------------------------ create data ------------------------------------
x_test = np.linspace(-1, 1, RESOLUTION).reshape(-1, 1)
y_test = f(x_test).squeeze()

# ---------------------------------- active learning ----------------------------------
model = BayesianPolynomialRegression(DEGREE, REGULARIZATION, NOISE)
test_error = np.zeros((REPEATS, ACQUISITIONS + 1))

tic = time.time()
for repeat in tqdm(range(REPEATS)):
    rng = np.random.default_rng(repeat)
    pool_x = np.empty((0, 1))
    pool_y = np.empty(0)

    # keep last history
    mean_history = np.zeros((ACQUISITIONS + 1, RESOLUTION))
    designs = []

    for step in range(ACQUISITIONS + 1):
        # rebuild a Latin hypercube of size n, reusing one pooled sample per stratum
        n = SAMPLES + step
        strata = ((pool_x[:, 0] + 1) / 2 * n).astype(int)
        keep = np.unique(strata, return_index=True)[1]
        empty = np.setdiff1d(np.arange(n), strata)

        x_new = (empty + rng.uniform(size=len(empty)))[:, None] / n * 2 - 1
        y_new = f(x_new).squeeze(-1) + rng.normal(0, NOISE, len(empty))
        pool_x = np.vstack([pool_x, x_new])
        pool_y = np.append(pool_y, y_new)

        x_train = np.vstack([pool_x[keep], x_new])
        y_train = np.append(pool_y[keep], y_new)
        designs.append((x_train, y_train))

        model.fit(x_train, y_train)
        mean_history[step] = model.forward(x_test)
        test_error[repeat, step] = np.sqrt(np.mean((mean_history[step] - y_test) ** 2))
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
samples = np.arange(SAMPLES, SAMPLES + ACQUISITIONS + 1)
mean_error = test_error.mean(axis=0)
x = x_test.squeeze()

# loss history
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.fill_between(
        samples, test_error.min(axis=0), test_error.max(axis=0), alpha=0.3, color="r"
    )
    ax.plot(samples, mean_error, "k")
    plt.show()

    # prediction histories
    fig, axs = plt.subplots(1, len(FRAMES), figsize=(3 * len(FRAMES), 3))
    for ax, step in zip(axs, FRAMES):
        ax.fill_between(
            x,
            mean_history[step] - 2 * NOISE,
            mean_history[step] + 2 * NOISE,
            alpha=0.4,
            color="b",
        )
        ax.plot(x, y_test, "k")
        ax.plot(x, mean_history[step], "b")
        ax.plot(*designs[step], "ro")

        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(step + SAMPLES)

    plt.show()

else:
    save_csv(
        CSV_DIR / "sine_active_random_history.csv",
        samples=samples,
        error=mean_error,
        error_p05=np.percentile(test_error, 5, axis=0),
        error_p95=np.percentile(test_error, 95, axis=0),
    )

    columns = {}
    for step in range(ACQUISITIONS + 1):
        columns[f"mean_{SAMPLES + step}"] = mean_history[step]
    save_csv(CSV_DIR / "sine_active_random.csv", x=x, **columns)

    save_csv(
        CSV_DIR / "sine_active_random_train.csv",
        x=np.concatenate([designs[step][0].squeeze(-1) for step in FRAMES]),
        y=np.concatenate([designs[step][1] for step in FRAMES]),
        samples=np.concatenate(
            [np.full(SAMPLES + step, SAMPLES + step) for step in FRAMES]
        ),
    )
