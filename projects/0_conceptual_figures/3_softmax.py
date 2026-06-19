import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from postprocessing import show_colorbar

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# matplotlib.rcParams["figure.dpi"] = 300
# matplotlib.rcParams["axes.linewidth"] = 2.5
# rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], "size": 22})
# rc("text", usetex=True)

# ----------------------------------- sample softmax ----------------------------------
x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)
x, y = np.meshgrid(x, y, indexing="ij")

z = np.exp(x) / (np.exp(x) + np.exp(y))

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
cb = ax.contourf(x, y, z, cmap="cividis", levels=64)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RESULTS_DIR / "softmax.png")
    plt.close()
else:
    plt.show()

show_colorbar(
    cb,
    path=RESULTS_DIR / "softmax_colorbar.png" if args.book else None,
    close=args.book,
    orientation="vertical",
)
