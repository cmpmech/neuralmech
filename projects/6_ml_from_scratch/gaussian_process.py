import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 32
NOISE = 0.14  # standard deviation, as in the sine examples of chapter 5
DRAWS = 5  # random functions drawn from prior and posterior
LENGTHS = (0.5, 2.0)  # length scales of conceptual prior draws with amplitude 1


# --------------------------------------- helper --------------------------------------
def kernel(a, b, length, scale):
    return scale**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / length**2)


def negative_log_likelihood(log_params):
    length, scale, noise = np.exp(log_params)
    K = kernel(x_train, x_train, length, scale) + noise**2 * np.eye(SAMPLES)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    return 0.5 * y_train @ alpha + np.log(np.diag(L)).sum()


# ------------------------------------ create data ------------------------------------
x_left = rng.uniform(-3, 0, SAMPLES // 2)
x_right = rng.uniform(1.5, 3, SAMPLES // 2)
x_train = np.concatenate([x_left, x_right])
y_train = np.sin(x_train) + NOISE * rng.standard_normal(SAMPLES)
x_test = np.linspace(-6, 6, 200)

# -------------------------------------- training -------------------------------------
length, scale, noise = np.exp(minimize(negative_log_likelihood, np.zeros(3)).x)
print(f"length {length:.2f}, scale {scale:.2f}, noise {noise:.3f}")

K = kernel(x_train, x_train, length, scale) + noise**2 * np.eye(SAMPLES)
K_star = kernel(x_train, x_test, length, scale)
mean = K_star.T @ np.linalg.solve(K, y_train)
variance = scale**2 - np.sum(K_star * np.linalg.solve(K, K_star), axis=0)
std = np.sqrt(np.clip(variance, 0, None))
std_total = np.sqrt(std**2 + noise**2)

K_test = kernel(x_test, x_test, length, scale)
covariance = K_test - K_star.T @ np.linalg.solve(K, K_star)
prior_draws = rng.multivariate_normal(np.zeros(len(x_test)), K_test, DRAWS, check_valid="ignore")
posterior_draws = rng.multivariate_normal(mean, covariance, DRAWS, check_valid="ignore")

x_grid = np.linspace(0, 8, 201)
x_inputs = np.sort(rng.uniform(0, 8, 10))
x_prior = np.concatenate([x_grid, x_inputs])
concept_draws = {}
for i, length_i in enumerate(LENGTHS):
    K_prior = kernel(x_prior, x_prior, length_i, 1)
    draws = rng.multivariate_normal(np.zeros(len(x_prior)), K_prior, 3, check_valid="ignore")
    for j, draw in enumerate(draws):
        concept_draws[f"l{i}_{j}"] = draw

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, ax = plt.subplots()
    ax.fill_between(x_test, mean - 2 * std_total, mean + 2 * std_total, alpha=0.2)
    ax.fill_between(x_test, mean - 2 * std, mean + 2 * std, color="b", alpha=0.3)
    ax.plot(x_test, mean, "b")
    ax.plot(x_test, np.sin(x_test), "k--")
    ax.plot(x_train, y_train, "ko")
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].plot(x_test, prior_draws.T)
    axs[1].plot(x_test, posterior_draws.T)
    axs[1].plot(x_train, y_train, "ko")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "gp_train.csv", x=x_train, y=y_train)
    save_csv(
        CSV_DIR / "gp.csv",
        x=x_test,
        mean=mean,
        std=std,
        std_total=std_total,
        y=np.sin(x_test),
    )
    save_csv(
        CSV_DIR / "gp_draws.csv",
        x=x_test,
        prior_std=np.full_like(x_test, scale),
        **{f"prior{i}": draw for i, draw in enumerate(prior_draws)},
        **{f"posterior{i}": draw for i, draw in enumerate(posterior_draws)},
    )
    grid_draws = {key: draw[: len(x_grid)] for key, draw in concept_draws.items()}
    input_draws = {key: draw[len(x_grid) :] for key, draw in concept_draws.items()}
    save_csv(CSV_DIR / "gp_prior_concept.csv", x=x_grid, **grid_draws)
    save_csv(CSV_DIR / "gp_prior_concept_inputs.csv", x=x_inputs, **input_draws)
