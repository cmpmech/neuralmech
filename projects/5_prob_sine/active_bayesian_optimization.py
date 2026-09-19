import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
SEED = 33
SAMPLES = 3  # initial random samples
ACQUISITIONS = 10  # bayesian optimization samples
RESOLUTION = 1000
NOISE = 0.05
KAPPA = 0.15  # exploration weight of the lower confidence bound

# postprocessing
FRAMES = [0, 1, 2, 3, 4, 5]

# model settings
DEGREE = 9
REGULARIZATION = 1e-2

# forrester function mapped onto [-1, 1]
f = lambda x: ((6 * (x + 1) / 2 - 2) ** 2 * np.sin(12 * (x + 1) / 2 - 4)) / 10

# ------------------------------------ create data ------------------------------------
x_test = np.linspace(-1, 1, RESOLUTION).reshape(-1, 1)
y_test = f(x_test).squeeze()

rng = np.random.default_rng(SEED)
x_train = rng.uniform(-1, 1, (SAMPLES, 1))
y_train = f(x_train).squeeze() + rng.normal(0, NOISE, SAMPLES)

# ------------------------------- bayesian optimization -------------------------------
model = BayesianPolynomialRegression(DEGREE, REGULARIZATION, NOISE)
mean_history = np.zeros((ACQUISITIONS + 1, RESOLUTION))
epistemic_history = np.zeros((ACQUISITIONS + 1, RESOLUTION))
acquisition_history = np.zeros((ACQUISITIONS + 1, RESOLUTION))
best_history = np.zeros(ACQUISITIONS + 1)

tic = time.time()
for step in range(ACQUISITIONS + 1):
    model.fit(x_train, y_train)
    mean_history[step] = model.forward(x_test)
    epistemic_history[step] = model.epistemic_std(x_test)
    acquisition_history[step] = mean_history[step] - KAPPA * epistemic_history[step]
    # noise free objective is the honest measure of what has been found so far
    best_history[step] = f(x_train).min()

    # sample where the surrogate promises the lowest value it could plausibly attain
    if step < ACQUISITIONS:
        x_new = x_test[acquisition_history[step].argmin()]
        x_train = np.vstack([x_train, x_new])
        y_train = np.append(y_train, f(x_new) + rng.normal(0, NOISE))
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
print(f"best: x={x_train[f(x_train).argmin(), 0]:.2e}, f={best_history[-1]:.2e}")

# ----------------------------------- postprocessing ----------------------------------
samples = np.arange(SAMPLES, SAMPLES + ACQUISITIONS + 1)
x = x_test.squeeze()

if not args.book:
    # convergence history
    fig, ax = plt.subplots()
    ax.plot(samples, best_history, "k")
    ax.plot(samples, np.full_like(samples, y_test.min(), dtype=float), "k--")
    plt.show()

    # surrogate and acquisition histories
    fig, axs = plt.subplots(2, len(FRAMES), figsize=(3 * len(FRAMES), 5))
    for ax, ax_acq, step in zip(axs[0], axs[1], FRAMES):
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

        idx = acquisition_history[step].argmin()
        ax_acq.plot(x, acquisition_history[step], "b")
        ax_acq.plot(x[idx], acquisition_history[step][idx], "ro")

        for axis in (ax, ax_acq):
            axis.set_xlim(-1, 1)
            axis.set_xticks([])
            axis.set_yticks([])
        ax.set_ylim(-1.5, 1.8)
        ax.set_title(step + SAMPLES)

    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    columns = {}
    for step in range(ACQUISITIONS + 1):
        columns[f"mean_{SAMPLES + step}"] = mean_history[step]
        columns[f"std_total_{SAMPLES + step}"] = np.sqrt(
            epistemic_history[step] ** 2 + NOISE**2
        )
        columns[f"acquisition_{SAMPLES + step}"] = acquisition_history[step]
    save_csv(CSV_DIR / "forrester_active_bo.csv", x=x, **columns)

    save_csv(
        CSV_DIR / "forrester_active_bo_train.csv",
        x=x_train.squeeze(),
        y=y_train,
        samples=np.maximum(np.arange(1, len(x_train) + 1), SAMPLES),
    )

    save_csv(
        CSV_DIR / "forrester_active_bo_history.csv",
        samples=samples,
        best=best_history,
    )
plt.close("all")
