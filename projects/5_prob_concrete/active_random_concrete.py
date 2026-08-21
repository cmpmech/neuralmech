import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config_concrete import load_concrete
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
SAMPLES = 10  # mixtures known before the first fit
ACQUISITIONS = 190  # mixtures acquired one at a time
REPEATS = 20  # a single random sequence is far too noisy to compare against

# model settings
DEGREE = 2
REGULARIZATION = 1e-2
NOISE = 6.0  # aleatoric standard deviation in mpa

# ------------------------------------ prepare data -----------------------------------
X, Y, standardizex = load_concrete()

# ---------------------------------- active learning ----------------------------------
# the same posterior as in the bayesian driver, but its covariance is never queried, so
# this is a plain regularized polynomial fit and the acquisition is the only difference
model = BayesianPolynomialRegression(DEGREE, REGULARIZATION, NOISE)
test_error = np.zeros((REPEATS, ACQUISITIONS))

tic = time.time()
for repeat in tqdm(range(REPEATS)):
    rng = np.random.default_rng(repeat)
    ids = rng.permutation(len(X))
    train_ids, pool_ids = list(ids[:SAMPLES]), list(ids[SAMPLES:])

    for step in range(ACQUISITIONS):
        model.fit(X[train_ids], Y[train_ids])
        test_error[repeat, step] = np.sqrt(np.mean((model.forward(X) - Y) ** 2))
        train_ids.append(pool_ids.pop(rng.integers(len(pool_ids))))
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
samples = np.arange(SAMPLES, SAMPLES + ACQUISITIONS)
mean_error = test_error.mean(axis=0)
std_error = test_error.std(axis=0)

fig, ax = plt.subplots()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_yscale("log")
ax.fill_between(
    samples, test_error.min(axis=0), test_error.max(axis=0), alpha=0.3, color="r"
)
ax.plot(samples, mean_error, "k")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig.savefig(RGB_PDF_DIR / "concrete_active_random.pdf")
    save_csv(
        CSV_DIR / "concrete_active_random.csv",
        samples=samples,
        error=mean_error,
        std=std_error,
    )
