import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.sparse import diags, eye, kron, vstack
from scipy.sparse.linalg import eigsh, spsolve

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# discretization
RESOLUTION = 64  # interior nodes per direction
DT = 1e-3
T = 0.5

# physics
BETA = 10.0  # conductivity k = 1 + BETA u
CENTER = (0.4, 0.6)
WIDTH = 0.15
INTENSITY = 80.0  # source peak, reached at T / 2

# reduced models
RANKS = np.arange(1, 31)
R = 8  # rank of the reduced fields shown
TIMES = [0.1, 0.175, 0.25, 0.325, 0.4]  # shown snapshots

# --------------------------------------- setup ---------------------------------------
n = RESOLUTION
h = 1 / (n + 1)
x = np.linspace(h, 1 - h, n)
x1, x2 = np.meshgrid(x, x, indexing="ij")
distance = (x1 - CENTER[0]) ** 2 + (x2 - CENTER[1]) ** 2
source = INTENSITY * np.exp(-distance / (2 * WIDTH**2)).flatten()
t = np.arange(round(T / DT) + 1) * DT
amplitude = np.sin(np.pi * t / T) ** 2

# differences across all cell faces, the boundary faces see the zero boundary value
d1 = diags([np.ones(n), -np.ones(n)], [0, -1], shape=(n + 1, n))
D = vstack([kron(d1, eye(n)), kron(eye(n), d1)]).tocsr() / h
A = abs(D) * h / 2  # temperature averaged onto the faces


def conductivity_matrix(u, beta):
    return D.T @ diags(1 + beta * (A @ u)) @ D


def simulate_full(beta):
    U = np.zeros((len(t), n * n))
    for tau in range(1, len(t)):
        lhs = eye(n * n) / DT + conductivity_matrix(U[tau - 1], beta)
        U[tau] = spsolve(lhs.tocsc(), U[tau - 1] / DT + amplitude[tau] * source)
    return U


def simulate_reduced(Phi, beta):
    DPhi = D @ Phi
    K0 = DPhi.T @ DPhi
    K1 = np.einsum("fa,fj,fb->jab", DPhi, A @ Phi, DPhi)  # conductivity is linear in u
    source_r = Phi.T @ source
    C = np.zeros((len(t), Phi.shape[1]))
    for tau in range(1, len(t)):
        lhs = np.eye(Phi.shape[1]) / DT + K0 + beta * np.tensordot(C[tau - 1], K1, 1)
        C[tau] = np.linalg.solve(lhs, C[tau - 1] / DT + amplitude[tau] * source_r)
    return C @ Phi.T


def relative_error(U_rom, U):
    return np.linalg.norm(U_rom - U) / np.linalg.norm(U)


# ---------------------------------- full order model ---------------------------------
tic = time.perf_counter()
U = simulate_full(BETA)
time_full = time.perf_counter() - tic
U_linear = simulate_full(0)

# ----------------------------------- reduced bases -----------------------------------
Phi_pod, S, _ = np.linalg.svd(U.T, full_matrices=False)
eigenvalues, Phi_modal = eigsh((D.T @ D).tocsc(), k=RANKS[-1], sigma=0)
Phi_modal = Phi_modal[:, np.argsort(eigenvalues)]  # modes of linearized problem

errors = {"pod": [], "modal": [], "modal_linear": []}
for rank in RANKS:
    U_pod = simulate_reduced(Phi_pod[:, :rank], BETA)
    U_modal = simulate_reduced(Phi_modal[:, :rank], BETA)
    U_modal_linear = simulate_reduced(Phi_modal[:, :rank], 0)
    errors["pod"].append(relative_error(U_pod, U))
    errors["modal"].append(relative_error(U_modal, U))
    errors["modal_linear"].append(relative_error(U_modal_linear, U_linear))

tic = time.perf_counter()
U_pod = simulate_reduced(Phi_pod[:, :R], BETA)
time_rom = time.perf_counter() - tic
U_modal = simulate_reduced(Phi_modal[:, :R], BETA)

print(f"{n * n} dofs, {len(t)} steps, max temperature {U.max():.2f}")
print(f"full {time_full:.2f} s, reduced (R = {R}) {time_rom:.3f} s")
for rank in RANKS:
    line = ", ".join(f"{k} {e[rank - 1]:.2e}" for k, e in errors.items())
    print(f"R = {rank:2d}: {line}")

# ----------------------------------- postprocessing ----------------------------------
idx = [round(s / DT) for s in TIMES]
fields = {}
errors_field = {}
for i, tau in enumerate(idx):
    fields[f"heat_full_{i}"] = U[tau]
    fields[f"heat_pod_{i}"] = U_pod[tau]
    fields[f"heat_modal_{i}"] = U_modal[tau]
    errors_field[f"heat_error_pod_{i}"] = np.abs(U_pod[tau] - U[tau])
    errors_field[f"heat_error_modal_{i}"] = np.abs(U_modal[tau] - U[tau])
vmax = max(e.max() for e in errors_field.values())
norm = LogNorm(vmin=vmax * 1e-5, vmax=vmax)
print(f"temperature 0 to {U.max():.3f}, error range {norm.vmin:.1e} to {vmax:.1e}")


def show_field(field, pad_mode, **kwargs):
    image = np.pad(field.reshape(n, n), 1, mode=pad_mode).T
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(image, origin="lower", interpolation="bilinear", **kwargs)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


for name, field in fields.items():
    fig = show_field(field, "constant", cmap="inferno", vmin=0, vmax=U.max())
    if args.book:
        fig.savefig(RGB_PDF_DIR / f"{name}.pdf", dpi=300)
    plt.close(fig)
for name, field in errors_field.items():
    fig = show_field(field + 1e-30, "edge", cmap="hot_r", norm=norm)  # boundary error is zero
    if args.book:
        fig.savefig(RGB_PDF_DIR / f"{name}.pdf", dpi=300)
    plt.close(fig)

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(RANKS, S[:30] / S[0], "k.")
    for name, color in zip(errors, ["r", "k", "b"]):
        axes[1].semilogy(RANKS, errors[name], color)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "rom_heat_error.csv", R=RANKS, **errors)
