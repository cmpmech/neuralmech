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
SAMPLES = 2  # initial random samples
ACQUISITIONS = 18  # active learning samples
REPEATS = 20
RESOLUTION = 1000
NOISE = 0.05

# postprocessing
FRAMES = [0, 2, 4, 6, 8, ACQUISITIONS]

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
max_epistemic = np.zeros((REPEATS, ACQUISITIONS + 1))

tic = time.time()
for repeat in tqdm(range(REPEATS)):
    rng = np.random.default_rng(repeat)
    x_train = rng.uniform(-1, 1, (SAMPLES, 1))
    y_train = f(x_train).squeeze() + rng.normal(0, NOISE, SAMPLES)

    # keep last history
    mean_history = np.zeros((ACQUISITIONS + 1, RESOLUTION))
    epistemic_history = np.zeros((ACQUISITIONS + 1, RESOLUTION))

    for step in range(ACQUISITIONS + 1):
        model.fit(x_train, y_train)
        mean_history[step] = model.forward(x_test)
        epistemic_history[step] = model.epistemic_std(x_test)
        test_error[repeat, step] = np.sqrt(np.mean((mean_history[step] - y_test) ** 2))
        max_epistemic[repeat, step] = epistemic_history[step].max()

        # find maximum epistemic uncertainty
        if step < ACQUISITIONS:
            x_new = x_test[epistemic_history[step].argmax()]
            x_train = np.vstack([x_train, x_new])
            y_train = np.append(y_train, f(x_new) + rng.normal(0, NOISE))
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
samples = np.arange(SAMPLES, SAMPLES + ACQUISITIONS + 1)
mean_error = test_error.mean(axis=0)
mean_epistemic = max_epistemic.mean(axis=0)
x = x_test.squeeze()

# loss history
if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.fill_between(
        samples, test_error.min(axis=0), test_error.max(axis=0), alpha=0.3, color="r"
    )
    ax.plot(samples, mean_error, "k")
    ax.plot(samples, mean_epistemic, "b")
    plt.show()

    # prediction histories
    fig, axs = plt.subplots(1, len(FRAMES), figsize=(3 * len(FRAMES), 3))
    for ax, step in zip(axs, FRAMES):
        total_std = np.sqrt(epistemic_history[step] ** 2 + NOISE**2)
        ax.fill_between(
            x,
            mean_history[step] - 2 * total_std,
            mean_history[step] + 2 * total_std,
            alpha=0.2,
            color="r",
        )
        ax.fill_between(
            x,
            mean_history[step] - 2 * NOISE,
            mean_history[step] + 2 * NOISE,
            alpha=0.4,
            color="b",
        )
        ax.plot(x, y_test, "k")
        ax.plot(x, mean_history[step], "b")
        ax.plot(x_train[: SAMPLES + step], y_train[: SAMPLES + step], "ro")

        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(step + SAMPLES)

    plt.show()

else:
    save_csv(
        CSV_DIR / "sine_active_bayesian_history.csv",
        samples=samples,
        error=mean_error,
        error_min=test_error.min(axis=0),
        error_max=test_error.max(axis=0),
        epistemic=mean_epistemic,
    )

    columns = {}
    for step in range(ACQUISITIONS + 1):
        columns[f"mean_{SAMPLES + step}"] = mean_history[step]
        columns[f"std_total_{SAMPLES + step}"] = np.sqrt(
            epistemic_history[step] ** 2 + NOISE**2
        )
    save_csv(CSV_DIR / "sine_active_bayesian.csv", x=x, **columns)

    save_csv(
        CSV_DIR / "sine_active_bayesian_train.csv",
        x=x_train.squeeze(),
        y=y_train,
        samples=np.maximum(np.arange(1, len(x_train) + 1), SAMPLES),
    )
