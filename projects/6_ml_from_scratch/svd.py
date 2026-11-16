import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
RANKS = [1, 5, 20, 50, 100]
MODES = [1, 2, 3]  # rank-1 terms shown separately

# ------------------------------------- load image ------------------------------------
img = Image.open(DATA_DIR / "images/sheep.jpg").convert("L")
img_arr = np.asarray(img, dtype=float) / 255


# ---------------------------- singular value decomposition ---------------------------
def svd(A):
    eigenvalues, U = np.linalg.eigh(A @ A.T)
    S = np.sqrt(np.clip(eigenvalues[::-1], 0, None))
    U = U[:, ::-1]
    Vt = U.T @ A / np.maximum(S, 1e-12)[:, None]
    return U, S, Vt


def reconstruct(rank):
    approx = U[:, :rank] @ (S[:rank, None] * Vt[:rank, :])
    return np.clip(approx, 0, 1)


U, S, Vt = svd(img_arr)
# U, S, Vt = np.linalg.svd(img_arr, full_matrices=False)

N, M = img_arr.shape
ratios = [N * M / (rank * (N + M + 1)) for rank in RANKS]
print(", ".join(f"rank {r}: {ratio:.1f}x" for r, ratio in zip(RANKS, ratios)))

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, len(RANKS), figsize=(4 * len(RANKS), 4 * N / M))
    for ax, rank, ratio in zip(axes, RANKS, ratios):
        ax.imshow(reconstruct(rank), cmap="gray")
        ax.set_title(f"rank {rank}, {ratio:.0f}x")
        ax.axis("off")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(S) + 1), S, "k")

    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "svd_singular_values.csv", idx=np.arange(1, len(S) + 1), value=S)
    for name, image in [("original", img_arr)] + [(r, reconstruct(r)) for r in RANKS]:
        fig, ax = plt.subplots(figsize=(M / 100, N / 100), dpi=100)
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"svd_rank_{name}.pdf")
        plt.close(fig)

    for i in MODES:
        mode = S[i - 1] * np.outer(U[:, i - 1], Vt[i - 1])
        limit = np.abs(mode).max()
        fig, ax = plt.subplots(figsize=(M / 100, N / 100), dpi=100)
        ax.imshow(mode, cmap="gray", vmin=-limit, vmax=limit)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PDF_DIR / f"svd_mode_{i}.pdf")
        plt.close(fig)
    columns = {f"u{i}": U[:, i - 1] for i in MODES}
    save_csv(CSV_DIR / "svd_mode_u.csv", row=np.arange(N), **columns)
    rows = {f"v{i}": Vt[i - 1] for i in MODES}
    save_csv(CSV_DIR / "svd_mode_v.csv", col=np.arange(M), **rows)
