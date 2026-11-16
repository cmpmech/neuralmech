import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from helper import beam_matrices, newmark
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# discretization
ELEMENTS = 100
DT = 1e-3
T = 20  # about 11 periods of the first mode
T_TRAIN = 4  # snapshots for the reduced basis come from this window only

# physics
PULSE = 0.02  # duration of the half-sine tip load, excites the first four modes

# reduced models
RANKS = np.arange(1, 11)
R = 3  # rank of the reduced model shown over time

# --------------------------------------- setup ---------------------------------------
K, M = beam_matrices(np.ones(ELEMENTS))
t = np.arange(round(T / DT) + 1) * DT
amplitude = np.where(t < PULSE, np.sin(np.pi * t / PULSE), 0)
shape = np.zeros(K.shape[0])
shape[-2] = 1  # tip deflection
x = np.linspace(0, 1, ELEMENTS + 1)


def deflection(U):
    return np.pad(U[..., 0::2], [(0, 0)] * (U.ndim - 1) + [(1, 0)])


def simulate_reduced(Phi):
    C = newmark(Phi.T @ M @ Phi, Phi.T @ K @ Phi, Phi.T @ shape, amplitude, DT)
    return C @ Phi.T


# ---------------------------------- full order model ---------------------------------
tic = time.perf_counter()
U = newmark(M, K, shape, amplitude, DT)
time_full = time.perf_counter() - tic

# ----------------------------------- reduced bases -----------------------------------
snapshots = U[t <= T_TRAIN].T
L = np.linalg.cholesky(M.toarray())  # mass inner product, as dofs include rotations
V, S, _ = np.linalg.svd(L.T @ snapshots, full_matrices=False)
Phi_pod = np.linalg.solve(L.T, V)
_, Phi_modal = eigh(K.toarray(), M.toarray(), subset_by_index=[0, RANKS[-1] - 1])

W_full = deflection(U)
errors = {}
for name, Phi in [("pod", Phi_pod), ("modal", Phi_modal)]:
    errors[name] = []
    for rank in RANKS:
        W = deflection(simulate_reduced(Phi[:, :rank]))
        errors[name].append(np.linalg.norm(W - W_full) / np.linalg.norm(W_full))

tic = time.perf_counter()
U_rom = simulate_reduced(Phi_pod[:, :R])
time_rom = time.perf_counter() - tic

print(f"{K.shape[0]} dofs, {len(t)} steps")
print(f"full {time_full:.2f} s, reduced (R = {R}) {time_rom:.2f} s")
for rank, e_pod, e_modal in zip(RANKS, errors["pod"], errors["modal"]):
    print(f"R = {rank:2d}: pod {e_pod:.2e}, modal {e_modal:.2e}")

# ----------------------------------- postprocessing ----------------------------------
modes_modal = deflection(Phi_modal[:, :4].T)
modes_modal /= modes_modal[:, -1:]  # unit tip deflection
modes_pod = deflection(Phi_pod[:, :4].T)
modes_pod *= np.sign(np.sum(modes_pod * modes_modal, axis=1, keepdims=True))
modes_pod /= np.abs(modes_pod).max(axis=1, keepdims=True)
modes_modal /= np.abs(modes_modal).max(axis=1, keepdims=True)
tip = W_full[:, -1]
tip_rom = deflection(U_rom)[:, -1]

if not args.book:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(x, modes_modal.T, "k--")
    axes[0, 0].plot(x, modes_pod.T, "r")
    axes[0, 1].semilogy(S[:20] / S[0], "k.")
    axes[1, 0].plot(t, tip, "k")
    axes[1, 0].plot(t, tip_rom, "r--")
    axes[1, 1].semilogy(RANKS, errors["pod"], "r")
    axes[1, 1].semilogy(RANKS, errors["modal"], "k")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    idx = np.arange(1, 21)
    save_csv(CSV_DIR / "rom_beam_singular_values.csv", idx=idx, value=S[:20] / S[0])
    save_csv(
        CSV_DIR / "rom_beam_modes.csv",
        x=x,
        **{f"pod{i + 1}": modes_pod[i] for i in range(4)},
        **{f"modal{i + 1}": modes_modal[i] for i in range(4)},
    )
    save_csv(CSV_DIR / "rom_beam_tip.csv", t=t[::10], full=tip[::10], rom=tip_rom[::10])
    save_csv(CSV_DIR / "rom_beam_error.csv", R=RANKS, **errors)
