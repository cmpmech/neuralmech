import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# ------------------------------- settings -------------------------------
n = 12
x = np.arange(n)
p = np.exp(-0.5 * ((x - 3) / 2.0) ** 2)
q = np.exp(-0.5 * ((x - 10) / 1.0) ** 2)
p /= p.sum()
q /= q.sum()

# -------------------------- optimal transport ---------------------------
n, m = len(p), len(q)

# cost matrix
D = np.abs(np.arange(n)[:, None] - np.arange(m)[None, :]).astype(float)
D /= D.max()

# optimal transport via LP
c = D.flatten()
A_eq = np.zeros((n + m, n * m))
for i in range(n):
    A_eq[i, i * m : (i + 1) * m] = 1
for j in range(m):
    A_eq[n + j, j::m] = 1
b_eq = np.concatenate([p, q])
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
Gamma = result.x.reshape(n, m)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(1, 1), dpi=n)
ax.imshow(Gamma.T, cmap="binary", origin="lower")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
# ------------------------- book postprocessing --------------------------
    fig.savefig(RESULTS_DIR / f"wasserstein_transport.png")
else:
    plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(1, 1), dpi=n)
ax.imshow(D.T, cmap="binary", origin="lower")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
# ------------------------- book postprocessing --------------------------
    fig.savefig(RESULTS_DIR / f"wasserstein_distance.png")
else:
    plt.show()
plt.close()

if args.book:
    # normalized for visualization
    save_csv(
        RESULTS_DIR / "wasserstein_distributions.csv", p=p / p.max(), q=q / q.max()
    )
else:
    fig, ax = plt.subplots()
    ax.plot(p, "k")
    ax.plot(q, "r")
    plt.show()
