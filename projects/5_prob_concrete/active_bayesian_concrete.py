import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config_concrete import features, load_concrete, sample_mixtures
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
SAMPLES = 10 # mixtures known before the first fit
ACQUISITIONS = 190 # mixtures acquired one at a time
REPEATS = 20 # averaged over the same initial sets as the random baseline
CANDIDATES = 20000 # mixtures drawn in the design box to probe the uncertainty

# model settings
DEGREE = 2
REGULARIZATION = 1e-2
NOISE = 6.0 # aleatoric standard deviation in mpa

# ------------------------------------ prepare data -----------------------------------
X, Y, standardizex = load_concrete()

# ---------------------------------- active learning ----------------------------------
model = BayesianPolynomialRegression(DEGREE, REGULARIZATION, NOISE)
max_epistemic = np.zeros((REPEATS, ACQUISITIONS))
test_error = np.zeros((REPEATS, ACQUISITIONS))

tic = time.time()
for repeat in tqdm(range(REPEATS)):
    rng = np.random.default_rng(repeat)
    ids = rng.permutation(len(X))
    train_ids, pool_ids = list(ids[:SAMPLES]), list(ids[SAMPLES:])

    for step in range(ACQUISITIONS):
        model.fit(X[train_ids], Y[train_ids])
        test_error[repeat, step] = np.sqrt(np.mean((model.forward(X) - Y) ** 2))

        # how badly the model extrapolates anywhere in the admissible design space
        candidates = standardizex(features(sample_mixtures(rng, CANDIDATES)))
        max_epistemic[repeat, step] = model.epistemic_std(candidates).max()

        # only mixtures the dataset contains can be cast, so the most uncertain one
        # among those is acquired
        train_ids.append(pool_ids.pop(model.epistemic_std(X[pool_ids]).argmax()))
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
samples = np.arange(SAMPLES, SAMPLES + ACQUISITIONS)
mean_error = test_error.mean(axis=0)
mean_epistemic = max_epistemic.mean(axis=0)

fig, ax = plt.subplots()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_yscale("log")
ax.plot(samples, mean_error, "k")
ax.plot(samples, mean_epistemic, "r")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig.savefig(RGB_PDF_DIR / "concrete_active_bayesian.pdf")
    save_csv(
        CSV_DIR / "concrete_active_bayesian.csv",
        samples=samples,
        error=mean_error,
        error_min=test_error.min(axis=0),
        error_max=test_error.max(axis=0),
        epistemic=mean_epistemic,
    )
