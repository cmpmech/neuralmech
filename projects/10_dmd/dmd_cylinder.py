import argparse
import time
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
R = 21  # rank, the mean plus ten oscillating pairs
WINDOWS = {"periodic": (40, 60), "transient": (2.5, 22.5)}  # training windows in D/U
MODES = 3  # oscillating mode pairs shown

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "cylinder_snapshots.npz")
vorticity, mask, t = data["vorticity"], data["mask"], data["t"]
X = vorticity.reshape(len(t), -1).T.astype(float)
dt = t[1] - t[0]


# --------------------------------------- helper --------------------------------------
def dmd(X1, X2, rank):
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    U, S, V = U[:, :rank], S[:rank], Vt[:rank].T
    A_tilde = U.T @ X2 @ V / S
    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = X2 @ V / S @ W
    amplitudes = np.linalg.lstsq(modes, X1[:, 0], rcond=None)[0]
    return eigenvalues, modes, amplitudes


def show_field(field, scale):
    field = np.ma.masked_array(field, mask)
    fig, ax = plt.subplots(figsize=(field.shape[0] / 50, field.shape[1] / 50), dpi=100)
    ax.imshow(field.T, origin="lower", cmap=cmr.guppy, vmin=-scale, vmax=scale)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


# ---------------------------------- train & forecast ---------------------------------
results = {}
for name, (start, end) in WINDOWS.items():
    ids = np.nonzero((t >= start) & (t < end))[0]
    tic = time.time()
    eigenvalues, modes, amplitudes = dmd(X[:, ids[:-1]], X[:, ids[1:]], R)
    toc = time.time()
    steps = np.arange(len(t)) - ids[0]
    X_pred = ((modes * amplitudes) @ eigenvalues[:, None] ** steps).real
    error = np.linalg.norm(X_pred - X, axis=0) / np.linalg.norm(X, axis=0)
    frequency = np.angle(eigenvalues) / (2 * np.pi * dt)
    weight = np.abs(amplitudes) * np.linalg.norm(modes, axis=0)
    results[name] = eigenvalues, modes, weight, frequency, X_pred, error, ids[0]
    print(f"{name}: dmd {toc - tic:.2f} s, {len(ids)} snapshots")
    strongest = np.sort(np.abs(frequency[np.argsort(-weight)[:5]]))
    print(f"  strongest frequencies {np.round(strongest, 3)}")
    radius = np.abs(eigenvalues)
    print(f"  |lambda| from {radius.min():.4f} to {radius.max():.4f}")
    print(f"  error at window end {error[ids[-1]]:.2e}, at end {error[-1]:.2e}")

S = np.linalg.svd(X[:, (t >= 40) & (t < 60)], compute_uv=False)
print(f"{X.shape[0]} degrees of freedom, {X.shape[1]} snapshots, dt = {dt:.3f} D/U")

# ----------------------------------- postprocessing ----------------------------------
_, modes, weight, frequency, *_ = results["periodic"]
order = np.argsort(-weight * (frequency > 1e-6))[:MODES]
fields = [modes[:, i].real.reshape(mask.shape) for i in order]
snapshot = X[:, -1].reshape(mask.shape)
scale = np.percentile(np.abs(snapshot), 99)  # the wall peaks would wash out the wake
predictions = [results[name][4][:, -1].reshape(mask.shape) for name in WINDOWS]

if not args.book:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(theta), np.sin(theta), "k", linewidth=0.5)
    for name, color in zip(WINDOWS, ["k", "r"]):
        lam = results[name][0]
        axes[0].plot(lam.real, lam.imag, color, marker="o", linestyle="")
        axes[1].semilogy(t, results[name][5], color)
        axes[2].plot(
            results[name][3], results[name][2], color, marker="o", linestyle=""
        )
    axes[0].set_aspect("equal")
    plt.show()
    for field in [snapshot, *predictions]:
        show_field(field, scale)
    for field in fields:
        show_field(field, np.abs(field).max())
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name in WINDOWS:
        eigenvalues, _, weight, frequency, _, error, start = results[name]
        save_csv(
            CSV_DIR / f"dmd_eigenvalues_{name}.csv",
            re=eigenvalues.real,
            im=eigenvalues.imag,
            frequency=frequency,
            weight=weight,
        )
        save_csv(CSV_DIR / f"dmd_error_{name}.csv", t=t[start:], error=error[start:])
    save_csv(
        CSV_DIR / "dmd_singular_values.csv",
        idx=np.arange(1, len(S) + 1),
        value=S / S[0],
    )
    show_field(snapshot, scale).savefig(RGB_PDF_DIR / "dmd_snapshot.pdf")
    for name, field in zip(WINDOWS, predictions):
        show_field(field, scale).savefig(RGB_PDF_DIR / f"dmd_forecast_{name}.pdf")
    for i, field in enumerate(fields):
        show_field(field, np.abs(field).max()).savefig(
            RGB_PDF_DIR / f"dmd_mode_{i + 1}.pdf"
        )
    plt.close("all")
