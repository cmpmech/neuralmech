import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

np.random.seed(0)

# -------------------------------------- settings -------------------------------------
DRAWS = 2000
X_INIT = 0
EPSILON = 0.1  # leapfrog step size
L = 20  # number of leapfrog steps
M = 1  # mass

# --------------------------------------- target --------------------------------------
MU = 2
SIGMA = 2
target = lambda x: (
    (1 / (SIGMA * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - MU) / SIGMA) ** 2)
)
U = lambda x: 0.5 * ((x - MU) / SIGMA) ** 2  # unnormalized potential
grad_U = lambda x: (x - MU) / SIGMA**2

# ---------------------------------------- HMC ----------------------------------------
samples = []
x_cur = X_INIT
count_prop = 0

tic = time.time()
for _ in range(DRAWS):
    rho_cur = np.random.normal(0, np.sqrt(M))

    x_prop, rho = x_cur, rho_cur
    for _ in range(L):
        rho = rho - 0.5 * EPSILON * grad_U(x_prop)
        x_prop = x_prop + EPSILON * rho / M
        rho = rho - 0.5 * EPSILON * grad_U(x_prop)

    H_cur = U(x_cur) + 0.5 * rho_cur**2 / M
    H_prop = U(x_prop) + 0.5 * rho**2 / M
    if np.random.rand() < np.exp(H_cur - H_prop):
        x_cur = x_prop
        count_prop += 1

    samples.append(x_cur)
toc = time.time()

print(f"accepted proposals: {count_prop}")
print(f"acceptance ratio: {count_prop / DRAWS:.4f}")
print(f"elapsed time {toc - tic:.4f} s")

samples = np.array(samples)
x = np.linspace(-6, 10, 200)

if not args.book:
    cutoff = 500
    fig, ax = plt.subplots()
    ax.plot(samples, "k")
    ax.plot(samples[:cutoff], "r")
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(samples[:cutoff], bins=50, density=True, alpha=0.3, color="b")
    ax.plot(x, target(x), "r--")
    ax.set_ylim(0, 0.3)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for cutoff in [100, 400, 2000]:
        save_csv(
            RESULTS_DIR / f"hmc_samples_{cutoff}.csv",
            s=np.arange(1, cutoff + 1),
            x=samples[:cutoff],
        )
    save_csv(RESULTS_DIR / "hmc_target.csv", x=x, y=target(x))
