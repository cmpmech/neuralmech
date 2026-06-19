import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DRAWS = 2000
TUNE = 10
CHAINS = 1
MU = 2
SIGMA = 2

# --------------------------------------- model ---------------------------------------
with pm.Model() as model:
    x = pm.Normal("x", mu=MU, sigma=SIGMA)
    trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, return_inferencedata=True)

print(az.summary(trace))

# ----------------------------------- postprocessing ----------------------------------
collected_samples = trace.posterior["x"].values.flatten()
x_range = np.linspace(-6, 10, 200)
target_pdf = lambda x: (
    (1 / (SIGMA * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - MU) / SIGMA) ** 2)
)

if not args.book:
    az.plot_trace(trace)
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(collected_samples, bins=50, density=True, alpha=0.3, color="b")
    ax.plot(x_range, target_pdf(x_range), "r--")
    plt.show()
