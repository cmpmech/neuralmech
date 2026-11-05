import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SIGMA = 0.5  # standard deviation of the gaussian policy
THRESHOLD = 1.0  # reward 1 for actions above, 0 below
SAMPLES = 10  # actions per gradient estimate
REPEATS = 10_000  # gradient estimates per policy mean, for their spread
RESOLUTION = 201

# ------------------------------------ create data ------------------------------------
mu = np.linspace(-1, 3, RESOLUTION)
expected_reward = norm.cdf((mu - THRESHOLD) / SIGMA)
gradient = norm.pdf((mu - THRESHOLD) / SIGMA) / SIGMA

actions = mu + SIGMA * rng.standard_normal((REPEATS, SAMPLES, RESOLUTION))
rewards = (actions > THRESHOLD).astype(float)
score = (actions - mu) / SIGMA**2
baseline = (rewards.sum(axis=1, keepdims=True) - rewards) / (SAMPLES - 1)
estimates = (rewards * score).mean(axis=1)
estimates_baseline = ((rewards - baseline) * score).mean(axis=1)

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(mu, (mu > THRESHOLD).astype(float), "b")
    axs[0].plot(mu, expected_reward, "k")
    axs[0].set_xlabel("action, policy mean")
    axs[1].plot(mu, gradient, "k")
    for est, color in [(estimates, "r"), (estimates_baseline, "b")]:
        mean, std = est.mean(axis=0), est.std(axis=0)
        axs[1].fill_between(mu, mean - std, mean + std, color=color, alpha=0.3)
    axs[1].set_xlabel("policy mean")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "policy_gradient_toy.csv",
        mu=mu,
        expected_reward=expected_reward,
        gradient=gradient,
        estimate_mean=estimates.mean(axis=0),
        estimate_std=estimates.std(axis=0),
        estimate_baseline_mean=estimates_baseline.mean(axis=0),
        estimate_baseline_std=estimates_baseline.std(axis=0),
        single_estimate=estimates[0],
    )
    print(f"largest std {estimates.std(axis=0).max():.2f} without baseline")
    print(f"largest std {estimates_baseline.std(axis=0).max():.2f} with baseline")
