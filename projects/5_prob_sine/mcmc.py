import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

np.random.seed(0)

# --------------------------- hyperparameters ----------------------------
NUM_SAMPLES = 2000
X_INIT = 0
PROPOSAL_STD = 1.0

# -------------------------------- target --------------------------------
MU = 2
SIGMA = 2
target = lambda x: (
    (1 / (SIGMA * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - MU) / SIGMA) ** 2)
)
unnormalized_target = lambda x: np.exp(-0.5 * ((x - MU) / SIGMA) ** 2)

# --------------------------------- MCMC ---------------------------------
samples = []
x_cur = X_INIT
count_prop = 0

tic = time.time()
for i in range(NUM_SAMPLES):
    x_prop = x_cur + np.random.normal(0, PROPOSAL_STD)
    p_cur = unnormalized_target(x_cur)
    p_prop = unnormalized_target(x_prop)
    if np.random.rand() < p_prop / p_cur:
        x_cur = x_prop
        count_prop += 1
    samples.append(x_cur)
toc = time.time()

print(f"accepted proposals: {count_prop}")
print(f"acceptance ratio: {count_prop / NUM_SAMPLES:.4f}")
print(f"elapsed time {toc - tic:.4f} s")

samples = np.array(samples)
x = np.linspace(-6, 10, 200)

if not args.book:
# ---------------------------- postprocessing ----------------------------
    cutoff = 2000
    fig, ax = plt.subplots()
    ax.plot(samples, "k")
    ax.plot(samples[:cutoff], "r")
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(samples[:cutoff], bins=50, density=True, alpha=0.3, color="b")
    ax.plot(x, target(x), "r--")
    ax.set_ylim(0, 0.3)
    plt.show()
else:
# ------------------------- book postprocessing --------------------------
    for cutoff in [100, 400, 2000]:
        save_csv(
            RESULTS_DIR / f"mcmc_samples_{cutoff}.csv",
            s=np.arange(1, cutoff + 1),
            x=samples[:cutoff],
        )
    save_csv(RESULTS_DIR / "mcmc_target.csv", x=x, y=target(x))
