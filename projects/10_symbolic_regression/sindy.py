import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from helper import polynomial_library, stlsq
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
# lorenz system
SIGMA = 10
RHO = 28
BETA = 8 / 3
X0 = [-8, 8, 27]
DT = 0.002
T = 10  # measured time window
T_PREDICT = 20

# sparse regression
DEGREE = 2  # of the polynomial library
THRESHOLD = 0.1

# failure study
NOISE_LEVELS = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]  # times state's spread
WINDOWS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]  # shorter measured time windows
WINDOW_NOISE = 1e-3
DEGREES = [2, 3, 5]

# pendulum
Q0 = 2.5  # initial angle, far outside small-angle range
Q0_TEST = 1.0  # unseen initial angle
PENDULUM_THRESHOLD = 0.05

NAMES = ["x", "y", "z"]
TRUTH = {
    "x": [-SIGMA, RHO, 0],
    "y": [SIGMA, -1, 0],
    "z": [0, 0, -BETA],
    "x z": [0, -1, 0],
    "x y": [0, 0, 1],
}


# --------------------------------------- helper --------------------------------------
def lorenz(t, s):
    x, y, z = s
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def simulate(f, s0, t):
    return solve_ivp(f, (t[0], t[-1]), s0, t_eval=t, rtol=1e-10, atol=1e-10).y.T


def identified(t, s):
    theta = polynomial_library(np.array([s]), DEGREE, NAMES)[0]
    return (theta @ xi)[0]


def true_coefficients(terms):
    return np.array([TRUTH.get(term, [0, 0, 0]) for term in terms], dtype=float)


def identify(X, degree):
    dXdt = np.gradient(X, DT, axis=0)
    theta, terms = polynomial_library(X, degree, NAMES)
    return stlsq(theta, dXdt, THRESHOLD), terms


def score(X, degree):
    xi, terms = identify(X, degree)
    xi_true = true_coefficients(terms)
    error = np.linalg.norm(xi - xi_true) / np.linalg.norm(xi_true)
    return error, int(np.sum((xi != 0) != (xi_true != 0)))


# ----------------------------------- lorenz system -----------------------------------
t = np.arange(0, T, DT)
X = simulate(lorenz, X0, t)
xi, terms = identify(X, DEGREE)
xi_true = true_coefficients(terms)

print(f"{'term':>6} " + " ".join(f"{'d' + n + '/dt':>9} {'true':>7}" for n in NAMES))
for term, row, row_true in zip(terms, xi, xi_true):
    print(f"{term:>6} " + " ".join(f"{a:9.4f} {b:7.3f}" for a, b in zip(row, row_true)))

t_predict = np.arange(0, T_PREDICT, DT)
X_true = simulate(lorenz, X0, t_predict)
X_sindy = simulate(identified, X0, t_predict)
deviation = np.linalg.norm(X_true - X_sindy, axis=1)
t_deviate = t_predict[np.argmax(deviation > 1)]
print(f"trajectories deviate by more than 1 after t = {t_deviate:.2f}")

# ----------------------------------- failure study -----------------------------------
noise_study = {degree: [] for degree in DEGREES}
for noise in NOISE_LEVELS:
    X_noisy = X + noise * X.std(axis=0) * rng.standard_normal(X.shape)
    for degree in DEGREES:
        noise_study[degree].append(score(X_noisy, degree))

window_study = {degree: [] for degree in DEGREES}
X_noisy = X + WINDOW_NOISE * X.std(axis=0) * rng.standard_normal(X.shape)
for window in WINDOWS:
    for degree in DEGREES:
        window_study[degree].append(score(X_noisy[: round(window / DT)], degree))

for degree in DEGREES:
    print(
        f"degree {degree}, noise levels: "
        + " ".join(f"{e:.1e}/{w}" for e, w in noise_study[degree])
    )
    print(
        f"degree {degree}, windows: "
        + " ".join(f"{e:.1e}/{w}" for e, w in window_study[degree])
    )

# -------------------------------------- pendulum -------------------------------------
pendulum = lambda t, s: [s[1], -np.sin(s[0])]
Q = simulate(pendulum, [Q0, 0], t)
dQdt = np.gradient(Q, DT, axis=0)
Q_test = simulate(pendulum, [Q0_TEST, 0], t)
libraries = {
    "polynomial": lambda Q: polynomial_library(Q, 3, ["q", "p"]),
    "with sine": lambda Q: (
        np.stack(
            [np.ones(len(Q)), Q[:, 0], Q[:, 1], np.sin(Q[:, 0]), np.cos(Q[:, 0])], 1
        ),
        ["1", "q", "p", "sin q", "cos q"],
    ),
}
for name, library in libraries.items():
    theta, pendulum_terms = library(Q)
    xi_pendulum = stlsq(theta, dQdt, PENDULUM_THRESHOLD)
    law = " ".join(
        f"{c:+.3f} {term}"
        for c, term in zip(xi_pendulum[:, 1], pendulum_terms)
        if c != 0
    )
    Q_pred = simulate(
        lambda t, s: (library(np.array([s]))[0] @ xi_pendulum)[0], [Q0_TEST, 0], t
    )
    error = np.abs(Q_pred - Q_test)[:, 0].max()
    print(f"pendulum, {name} library: dp/dt = {law}")
    print(f"  max angle error from unseen q0 = {Q0_TEST}: {error:.2e}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(t_predict, X_true[:, 0], "k--")
    axes[0].plot(t_predict, X_sindy[:, 0], "b")
    axes[1].plot(X_true[:, 0], X_true[:, 2], "k--", linewidth=0.5)
    axes[1].plot(X_sindy[:, 0], X_sindy[:, 2], "b", linewidth=0.5)
    for degree, color in zip(DEGREES, ["k", "b", "r"]):
        axes[2].loglog(
            NOISE_LEVELS[1:], [e for e, _ in noise_study[degree][1:]], color, marker="o"
        )
        axes[3].loglog(WINDOWS, [e for e, _ in window_study[degree]], color, marker="o")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    xi_true = true_coefficients(terms)
    columns = {f"d{n}dt": xi[:, i] for i, n in enumerate(NAMES)}
    columns |= {f"d{n}dt_true": xi_true[:, i] for i, n in enumerate(NAMES)}
    save_csv(
        CSV_DIR / "sindy_lorenz_coefficients.csv",
        term=[t.replace(" ", "") for t in terms],
        **columns,
    )
    every = 5
    save_csv(
        CSV_DIR / "sindy_lorenz_trajectory.csv",
        t=t_predict[::every],
        x=X_true[::every, 0],
        z=X_true[::every, 2],
        x_sindy=X_sindy[::every, 0],
        z_sindy=X_sindy[::every, 2],
    )
    for study, values, name in [
        (noise_study, NOISE_LEVELS, "noise"),
        (window_study, WINDOWS, "window"),
    ]:
        columns = {f"error_{d}": [e for e, _ in study[d]] for d in DEGREES}
        columns |= {f"wrong_{d}": [w for _, w in study[d]] for d in DEGREES}
        save_csv(CSV_DIR / f"sindy_lorenz_{name}.csv", **{name: values}, **columns)
