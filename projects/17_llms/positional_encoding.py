import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
N = 100  # sequence length
D = 200  # embedding dimension
BASE = 20  # 10000 in practice, small here to show all frequencies within N positions

# ------------------------------------ create data ------------------------------------
n = np.arange(N)[:, None]
i = np.arange(D)[None, :]
angle = n / BASE ** (i / D)
E = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))  # (N, D)

# dimensions on the vertical axis, positions on the horizontal axis
encodings = {
    "pos_encoding_even": E[:, 0::2].T,
    "pos_encoding_odd": E[:, 1::2].T,
    "pos_encoding": E.T,
}

# ----------------------------------- postprocessing ----------------------------------
figures = {}
for name, image in encodings.items():
    figures[name] = plt.figure(figsize=(4, 4))
    ax = figures[name].add_axes([0, 0, 1, 1])
    ax.imshow(
        image, cmap="cividis", vmin=-1, vmax=1, aspect="auto", interpolation="none"
    )
    ax.axis("off")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, fig in figures.items():
        fig.savefig(RGB_PDF_DIR / f"{name}.pdf")
