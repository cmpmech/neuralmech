import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["axes.linewidth"] = 2.5
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], "size": 22})
rc("text", usetex=True)

# --------------------------- softmax sampling ---------------------------
x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)
x, y = np.meshgrid(x, y, indexing="ij")

z = np.exp(x) / (np.exp(x) + np.exp(y))

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
cb = ax.pcolormesh(x, y, z, cmap="cividis")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
if args.book:
    plt.savefig(RESULTS_DIR / "softmax.png", bbox_inches="tight", pad_inches=0)
    plt.close()
else:
    plt.show()

fig, ax = plt.subplots(figsize=(0.7, 3), dpi=100)
ax.set_visible(False)
cbar = fig.colorbar(
    cb, ax=ax, fraction=1.0, pad=0.04, format="%.1f", location="right", aspect=15
)
cbar.ax.tick_params(width=1)
cbar.outline.set_visible(False)
fig.tight_layout(pad=0)
ax.set_rasterized(True)
if args.book:
    plt.savefig(
        RESULTS_DIR / "softmax_colorbar.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
else:
    plt.show()
