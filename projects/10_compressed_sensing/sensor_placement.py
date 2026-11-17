import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
TRAIN = (40, 60)  # periodic window for the basis, the remaining snapshots are tested
SENSORS = np.arange(2, 31)  # number of sensors, equal to the number of basis vectors
SHOWN = 10  # sensors in the shown reconstruction
DRAWS = 50  # random sensor sets per number of sensors

rng = np.random.default_rng(0)

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "cylinder_snapshots.npz")  # from 10_dmd/cylinder_snapshots.py
vorticity, mask, t = data["vorticity"], data["mask"], data["t"]
X = vorticity.reshape(len(t), -1).T.astype(float)
X_train = X[:, (t >= TRAIN[0]) & (t < TRAIN[1])]
X_test = X[:, t >= TRAIN[1]]
fluid = np.nonzero(~mask.flatten())[0]

mean = X_train.mean(axis=1, keepdims=True)
U = np.linalg.svd(X_train - mean, full_matrices=False)[0]


# --------------------------------------- helper --------------------------------------
def qr_sensors(Phi):
    return qr(Phi.T, pivoting=True, mode="economic")[2][: Phi.shape[1]]


def reconstruct(Phi, sensors):
    c = np.linalg.lstsq(Phi[sensors], X_test[sensors] - mean[sensors], rcond=None)[0]
    return mean + Phi @ c


def error(X_pred):
    return np.linalg.norm(X_pred - X_test) / np.linalg.norm(X_test)


# ------------------------------------ reconstruction ---------------------------------
errors = {"qr": [], "random": [], "projection": []}
for p in SENSORS:
    Phi = U[:, :p]
    errors["qr"].append(error(reconstruct(Phi, qr_sensors(Phi))))
    draws = [error(reconstruct(Phi, rng.choice(fluid, p, replace=False))) for _ in range(DRAWS)]
    errors["random"].append(np.median(draws))
    errors["projection"].append(error(mean + Phi @ (Phi.T @ (X_test - mean))))
for p, e_qr, e_random, e_proj in zip(SENSORS, *errors.values()):
    if p in [5, 10, 20, 30]:
        print(f"{p} sensors: qr {e_qr:.3f}, random {e_random:.3f}, projection {e_proj:.3f}")

Phi = U[:, :SHOWN]
sensors_qr = qr_sensors(Phi)
sensors_random = rng.choice(fluid, SHOWN, replace=False)
fields = {
    "snapshot": X_test[:, -1],
    "qr": reconstruct(Phi, sensors_qr)[:, -1],
    "random": reconstruct(Phi, sensors_random)[:, -1],
}
scale = np.percentile(np.abs(fields["snapshot"]), 99)  # as in dmd_cylinder.py


def show_field(field):
    field = np.ma.masked_array(field.reshape(mask.shape), mask)
    fig, ax = plt.subplots(figsize=(field.shape[0] / 50, field.shape[1] / 50), dpi=100)
    ax.imshow(field.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    for name, field in fields.items():
        fig = show_field(field)
        i, j = np.unravel_index(sensors_qr if name != "random" else sensors_random, mask.shape)
        fig.axes[0].plot(i, j, "k.")
    fig, ax = plt.subplots()
    for e in errors.values():
        ax.semilogy(SENSORS, e)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, field in fields.items():
        show_field(field).savefig(RGB_PDF_DIR / f"sensors_{name}.pdf")
    plt.close("all")
    for name, sensors in [("qr", sensors_qr), ("random", sensors_random)]:
        i, j = np.unravel_index(sensors, mask.shape)
        save_csv(CSV_DIR / f"sensors_{name}.csv", x=(i + 0.5) / mask.shape[0], y=(j + 0.5) / mask.shape[1])
    save_csv(CSV_DIR / "sensors_error.csv", p=SENSORS, **errors)
