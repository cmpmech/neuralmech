import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper import stlsq
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
# geometry
ELEMENTS = 20
TAPER = 0.25  # cross-section at loaded end relative to support

# material, sigma = E eps + Q eps^3
E = 1.0
Q = 20.0

# measurements
LOADS = [0.05, 0.1, 0.15, 0.2, 0.25]  # end forces of load steps
NOISE = 1e-4  # displacement noise relative to largest displacement

# sparse regression
THRESHOLD = 0.5
LIBRARY = {
    "eps": lambda eps: eps,
    "eps^2": lambda eps: eps**2,
    "eps^3": lambda eps: eps**3,
}
LARGE_LIBRARY = LIBRARY | {"eps^4": lambda eps: eps**4, "eps^5": lambda eps: eps**5}

x = np.linspace(0, 1, ELEMENTS + 1)
h = 1 / ELEMENTS
x_mid = (x[:-1] + x[1:]) / 2


# --------------------------------------- helper --------------------------------------
def assemble(element_forces):
    return np.concatenate(
        [
            -element_forces[:1],
            element_forces[:-1] - element_forces[1:],
            element_forces[-1:],
        ]
    )


def solve(A, F):
    u = np.zeros(ELEMENTS + 1)
    for _ in range(50):
        eps = np.diff(u) / h
        residual = assemble(A * (E * eps + Q * eps**3))[1:]
        residual[-1] -= F
        if np.linalg.norm(residual) < 1e-12:
            break
        k = A * (E + 3 * Q * eps**2) / h
        K = np.diag(np.append(k[1:], 0) + k) - np.diag(k[1:], 1) - np.diag(k[1:], -1)
        u[1:] -= np.linalg.solve(K, residual)
    return u


def measure(A, loads):
    displacements = []
    for F in loads:
        u = solve(A, F)
        u[1:] += NOISE * np.abs(u).max() * rng.standard_normal(ELEMENTS)
        displacements.append(u)
    return displacements


def identify(A, loads, displacements, library):
    rows, rhs = [], []
    for F, u in zip(loads, displacements):
        eps = np.diff(u) / h
        rows.append(np.stack([assemble(A * g(eps)) for g in library.values()], axis=1))
        rhs.append(
            np.concatenate([[-F], np.zeros(ELEMENTS - 1), [F]])
        )  # reaction, free nodes, load
    M = np.concatenate(rows)
    singular_values = np.linalg.svd(M, compute_uv=False)
    theta = stlsq(M, np.concatenate(rhs)[:, None], THRESHOLD)[:, 0]
    return theta, singular_values


def stress(theta, library, eps):
    return sum(t * g(eps) for t, g in zip(theta, library.values()))


# ------------------------------------ measurements -----------------------------------
A_tapered = 1 - (1 - TAPER) * x_mid
A_prismatic = np.ones(ELEMENTS)
tapered = measure(A_tapered, LOADS)
prismatic = measure(A_prismatic, LOADS[-1:])

# ----------------------------------- identification ----------------------------------
cases = {
    "tapered": (A_tapered, LOADS, tapered, LIBRARY),
    "prismatic, one load step": (A_prismatic, LOADS[-1:], prismatic, LIBRARY),
    "tapered, large library": (A_tapered, LOADS, tapered, LARGE_LIBRARY),
}
eps_max = np.diff(tapered[-1]).max() / h
eps_grid = np.linspace(0, eps_max, 100)
curves = {"truth": E * eps_grid + Q * eps_grid**3}
for case, (A, loads, displacements, library) in cases.items():
    theta, singular_values = identify(A, loads, displacements, library)
    curves[case] = stress(theta, library, eps_grid)
    law = " ".join(f"{t:+.3f} {name}" for t, name in zip(theta, library) if t != 0)
    print(f"{case}: sigma = {law}")
    print("  singular values " + " ".join(f"{s:.1e}" for s in singular_values))
    print(f"  condition number {singular_values[0] / singular_values[-1]:.1e}")
eps_min = np.diff(tapered[0]).min() / h
print(f"measured strains of tapered bar from {eps_min:.3f} to {eps_max:.3f}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, (ax_u, ax_sigma) = plt.subplots(1, 2, figsize=(10, 4))
    for u in tapered:
        ax_u.plot(x, u, "ko", markerfacecolor="none")
    ax_sigma.plot(eps_grid, curves["truth"], "k--")
    for curve, color in zip(list(curves.values())[1:], ["b", "r", "g"]):
        ax_sigma.plot(eps_grid, curve, color)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    names = ["truth", "tapered", "prismatic", "large_library"]
    save_csv(
        CSV_DIR / "euclid_stress_strain.csv",
        eps=eps_grid,
        **dict(zip(names, curves.values())),
    )
    save_csv(
        CSV_DIR / "euclid_displacements.csv",
        x=x,
        u=tapered[-1],
        u_true=solve(A_tapered, LOADS[-1]),
    )
