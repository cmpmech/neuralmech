import argparse
from fractions import Fraction
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 100
NOISE = 1e-3  # relative measurement noise of the target
EXPONENTS = np.arange(-2, 2.5, 0.5)  # grammar of the brute-force search
TOLERANCE = 1e-2  # relative distance at which a known constant is accepted

# units as exponents of mass, length, and time
UNITS = {
    "F": [1, 1, -2],
    "f": [0, 0, -1],
    "E": [1, -1, -2],
    "rho": [1, -3, 0],
    "L": [0, 1, 0],
    "I": [0, 4, 0],
    "A": [0, 2, 0],
}

# log10 ranges of the sampled inputs, in SI units
RANGES = {"E": (9, 11.5), "rho": (3, 4), "L": (0, 1), "I": (-8, -5), "A": (-3, -1)}

CONSTANTS = {
    "1": 1,
    "2": 2,
    "1/2": 1 / 2,
    "pi": np.pi,
    "pi^2": np.pi**2,
    "2 pi": 2 * np.pi,
    "1/pi": 1 / np.pi,
    "1/(2 pi)": 1 / (2 * np.pi),
    "sqrt(2)": np.sqrt(2),
}

# target and inputs
PROBLEMS = {
    "buckling": ("F", ["E", "L", "I"]),
    "cantilever": ("f", ["E", "rho", "L", "I", "A"]),
    "cantilever without rho": ("f", ["E", "L", "I", "A"]),
}
LAWS = {
    "F": lambda v: np.pi**2 * v["E"] * v["I"] / v["L"] ** 2,
    "f": lambda v: (
        1.875104**2
        / (2 * np.pi)
        * np.sqrt(v["E"] * v["I"] / (v["rho"] * v["A"] * v["L"] ** 4))
    ),
}


# --------------------------------------- helper --------------------------------------
def dimensionless_groups(target, inputs):
    D = np.array([UNITS[name] for name in inputs]).T
    basis = []
    for i in range(len(inputs)):
        if np.linalg.matrix_rank(D[:, basis + [i]]) > len(basis):
            basis.append(i)
    p, residual = np.linalg.lstsq(D[:, basis], UNITS[target], rcond=None)[:2]
    if residual.size and residual[0] > 1e-12:
        return None  # target units are not a product of input units
    target_group = {target: 1} | {inputs[i]: -e for i, e in zip(basis, p)}
    input_groups = []
    for j in [j for j in range(len(inputs)) if j not in basis]:
        q = np.linalg.lstsq(D[:, basis], D[:, j], rcond=None)[0]
        input_groups.append({inputs[j]: 1} | {inputs[i]: -e for i, e in zip(basis, q)})
    return target_group, input_groups


def evaluate(group, values):
    return np.prod([values[name] ** e for name, e in group.items()], axis=0)


def brute_force(y, X):
    best = (np.inf, None, None)
    for exponents in product(EXPONENTS, repeat=X.shape[1]):
        residual = np.log(y) - np.log(X) @ exponents
        error = np.std(residual)  # relative error of the best constant
        if error < best[0]:
            best = (error, exponents, np.exp(residual.mean()))
    return best


def name_group(group):
    return " ".join(
        f"{name}^{Fraction(e).limit_denominator(4)}"
        for name, e in group.items()
        if abs(e) > 1e-12
    )


def name_constant(c):
    for name, value in CONSTANTS.items():
        if abs(c / value - 1) < TOLERANCE:
            return name
    return "none"


# ------------------------------------ create data ------------------------------------
values = {name: 10 ** rng.uniform(*RANGES[name], SAMPLES) for name in RANGES}
for target, law in LAWS.items():
    values[target] = law(values) * (1 + NOISE * rng.standard_normal(SAMPLES))

# -------------------------------- dimensional analysis -------------------------------
groups = {}
for problem, (target, inputs) in PROBLEMS.items():
    D = np.array([UNITS[name] for name in [target] + inputs]).T
    rank = np.sum(np.linalg.svd(D, compute_uv=False) > 1e-10)
    variables = len(inputs) + 1
    print(f"{problem}: {variables} variables, rank {rank}, {variables - rank} groups")
    groups[problem] = dimensionless_groups(target, inputs)

# ------------------------------------ brute force ------------------------------------
results = {}
for problem, (target, inputs) in PROBLEMS.items():
    raw_laws = len(EXPONENTS) ** len(inputs)
    print(problem)
    if groups[problem] is None:
        X = np.stack([values[name] for name in inputs], axis=1)
        error, exponents, c = brute_force(values[target], X)
        print(f"  no group contains {target}, raw search over {raw_laws} laws")
        print(
            f"  best raw law {c:.3e} "
            + " ".join(f"{n}^{Fraction(e)}" for n, e in zip(inputs, exponents))
        )
        print(f"  relative error {error:.2e}")
        continue
    target_group, input_groups = groups[problem]
    y = evaluate(target_group, values)
    X = np.stack([evaluate(group, values) for group in input_groups], axis=1)
    error, exponents, c = brute_force(y, X)
    results[problem] = (y, np.prod(X**exponents, axis=1))
    law = " ".join(
        f"({name_group(g)})^{Fraction(e)}" for g, e in zip(input_groups, exponents)
    )
    print(f"  search over {len(EXPONENTS) ** X.shape[1]} laws instead of {raw_laws}")
    print(f"  {name_group(target_group)} = {c:.4f} {law}")
    print(f"  constant matches {name_constant(c)}, relative error {error:.2e}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, (y, x) in zip(axes, results.values()):
        ax.loglog(x, y, "ko", markerfacecolor="none")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for problem, (y, x) in results.items():
        save_csv(CSV_DIR / f"ai_feynman_{problem}.csv", x=x, y=y)
    save_csv(CSV_DIR / "ai_feynman_buckling_raw.csv", L=values["L"], F=values["F"])
