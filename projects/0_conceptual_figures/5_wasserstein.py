import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
N = 12

x = np.arange(N)
p = np.exp(-0.5 * ((x - 3) / 2.0) ** 2)
q = np.exp(-0.5 * ((x - 10) / 1.0) ** 2)
p /= p.sum()
q /= q.sum()

# --------------------------------- optimal transport ---------------------------------
n, m = len(p), len(q)  # here n=m

# cost matrix
D = np.abs(np.arange(n)[:, None] - np.arange(m)[None, :]).astype(float)
D /= D.max()

# optimal transport via linear programming
d = D.flatten()  # lp uses vectors (not matrices)
A_eq = np.zeros((n + m, n * m))  # n + m constraints for the n * m dofs (flat Gamma)
# first n rows (summed) must equal p[i]
for i in range(n):
    A_eq[i, i * m : (i + 1) * m] = 1
# last m rows (summed) must equal q[j]
for j in range(m):
    A_eq[n + j, j::m] = 1
# rhs of constraints
b_eq = np.concatenate([p, q])

result = linprog(d, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
Gamma = result.x.reshape(n, m)

# -------------------------------- book postprocessing --------------------------------
fig, ax = plt.subplots(figsize=(1, 1), dpi=n)
ax.imshow(Gamma.T, cmap="binary", origin="lower")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    fig.savefig(RGB_PDF_DIR / f"wasserstein_transport.pdf")
else:
    plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(1, 1), dpi=n)
ax.imshow(D.T, cmap="binary", origin="lower")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    fig.savefig(RGB_PDF_DIR / f"wasserstein_distance.pdf")
else:
    plt.show()
plt.close()

if args.book:
    # normalized for visualization
    scale = (p / p.max()).sum() / (q / q.max()).sum()
    save_csv(
        CSV_DIR / "wasserstein_distributions.csv", p=p / p.max(), q=q / q.max() * scale
    )
else:
    fig, ax = plt.subplots()
    ax.plot(p, "k")
    ax.plot(q, "r")
    plt.show()
