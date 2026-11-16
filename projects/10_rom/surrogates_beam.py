import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.polynomial.legendre import legvander2d
from numpy.polynomial.polynomial import polyvander2d
from scipy.linalg import eigh
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from helper import beam_matrices
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
SAMPLES = 20
ELEMENTS = 20
BOUNDS = (0.1, 1.9)  # bending stiffness range of each beam half
RESOLUTION = 128
MC_SAMPLES = 10_000


# --------------------------------------- helper --------------------------------------
def frequency(xi):
    EI = np.repeat(BOUNDS[0] + (xi + 1) / 2 * (BOUNDS[1] - BOUNDS[0]), ELEMENTS // 2)
    K, M = beam_matrices(EI)
    K, M = K.toarray(), M.toarray()
    eigenvalues = eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])
    return np.sqrt(eigenvalues[0])


def features(xi, vander, degree):
    i, j = np.meshgrid(np.arange(degree + 1), np.arange(degree + 1), indexing="ij")
    return vander(xi[:, 0], xi[:, 1], [degree, degree])[:, (i + j <= degree).flatten()]


def polynomial_fit(vander, degree):
    coefficients = np.linalg.lstsq(features(xi, vander, degree), y, rcond=None)[0]
    return coefficients, lambda xi_new: features(xi_new, vander, degree) @ coefficients


# ------------------------------------ create data ------------------------------------
# both stiffnesses are uniform random variables, mapped to xi in [-1, 1]
xi = 2 * qmc.LatinHypercube(d=2, seed=0).random(SAMPLES) - 1
y = np.array([frequency(sample) for sample in xi])

grid = np.linspace(-1, 1, RESOLUTION)
xi1, xi2 = np.meshgrid(grid, grid, indexing="ij")
xi_test = np.stack([xi1.flatten(), xi2.flatten()], axis=1)
y_test = np.array([frequency(sample) for sample in xi_test])

xi_mc = rng.uniform(-1, 1, (MC_SAMPLES, 2))
y_mc = np.array([frequency(sample) for sample in xi_mc])

# ------------------------------------- surrogates ------------------------------------
_, response_surface = polynomial_fit(polyvander2d, 2)

kernel = ConstantKernel() * RBF(length_scale=[1.0, 1.0])
gp = GaussianProcessRegressor(
    kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0
)
gp.fit(xi, y)

coefficients, chaos = polynomial_fit(legvander2d, 3)

# mean and variance follow from orthogonality, E[P_i^2] = 1 / (2 i + 1) for uniform xi
i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
keep = (i + j <= 3).flatten()
norms = 1 / ((2 * i + 1) * (2 * j + 1)).flatten()[keep]
chaos_mean = coefficients[0]
chaos_std = np.sqrt(np.sum(coefficients[1:] ** 2 * norms[1:]))

# ----------------------------------- postprocessing ----------------------------------
surrogates = {"response_surface": response_surface, "kriging": gp.predict}
surrogates["chaos"] = chaos
errors = {name: np.abs(f(xi_test) - y_test) for name, f in surrogates.items()}
vmax = max(error.max() for error in errors.values())
norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)

print(f"frequency {y_test.min():.3f} to {y_test.max():.3f}")
print(f"error range {norm.vmin:.1e} to {vmax:.1e}")
for name, f in surrogates.items():
    y_pred = f(xi_mc)
    relative = np.linalg.norm(f(xi_test) - y_test) / np.linalg.norm(y_test)
    statistics = f"mean {y_pred.mean():.4f}, std {y_pred.std():.4f}"
    print(f"{name}: relative error {relative:.2e}, {statistics}")
print(f"chaos coefficients: mean {chaos_mean:.4f}, std {chaos_std:.4f}")
print(f"monte carlo: mean {y_mc.mean():.4f}, std {y_mc.std():.4f}")


def show_field(field, **kwargs):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(field.reshape(RESOLUTION, RESOLUTION).T, origin="lower", **kwargs)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


fig = show_field(y_test, cmap="cividis")
if args.book:
    fig.savefig(RGB_PDF_DIR / "surrogate_ground_truth.pdf", dpi=300)
for name, error in errors.items():
    fig = show_field(error + 1e-30, cmap="hot_r", norm=norm)
    if args.book:
        fig.savefig(RGB_PDF_DIR / f"surrogate_error_{name}.pdf", dpi=300)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    x1, x2 = (xi.T + 1) / 2  # coordinates in the unit square of the images
    save_csv(CSV_DIR / "surrogate_samples.csv", x=x1, y=x2)
