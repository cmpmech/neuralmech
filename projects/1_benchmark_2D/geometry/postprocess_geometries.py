import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).parent
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries").resolve()
RESULTS_DIR = (BASE_DIR / "../../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
SOURCE = "ctscans"
# SOURCE = "sandstones11"
# SOURCE = "ibm"
# SOURCE = "presalt"
# SOURCE = "mrccm"
# SOURCE = "deeprocksr"
# SOURCE = "cfrp_twill"
# SOURCE = "alsi10mg"
# SOURCE = "cement"
# SOURCE = "kfoam"
# SOURCE = "syntactic_foam"
# SOURCE = "gfrp_ud"
# SOURCE = "cfrp_t700"
RESOLUTION = 256  # 128, 256
THRESHOLD = 0.5  # binarization of the gray value scaled to [0, 1]

# book figure: (source, material type, sample), rocks first, then engineered materials
BOOK_SAMPLES = [
    ("ctscans", "B_HAI", 0),
    ("ctscans", "DD", 4),
    ("sandstones11", "Bentheimer", 0),
    ("ibm", "kocurek_13a", 0),
    ("presalt", "SW04", 0),
    ("mrccm", "MEC", 3),
    ("deeprocksr", "sandstone", 1),
    ("cfrp_twill", "cfrp_twill", 19),
    ("cfrp_t700", "cfrp_t700", 0),
    ("gfrp_ud", "gfrp_srct", 2),
    ("alsi10mg", "alsi10mg", 3),
    ("cement", "portland_calcite", 0),
    ("kfoam", "graphite_foam", 0),
    ("syntactic_foam", "syntactic_13_10", 2),
]

# ------------------------------------- load data -------------------------------------
if not args.book:
    files = sorted((GEOMETRY_DIR / SOURCE).glob(f"*_{RESOLUTION}.pt"))
    kinds = [file.stem.rsplit("_", 1)[0] for file in files]
    images = [
        torch.load(file, weights_only=False, map_location=device)[0] for file in files
    ]
else:
    images = []
    for source, kind, i in BOOK_SAMPLES:
        file = GEOMETRY_DIR / source / f"{kind}_{RESOLUTION}.pt"
        images.append(torch.load(file, weights_only=False, map_location=device)[i])

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    cols = int(np.ceil(np.sqrt(len(files))))
    rows = int(np.ceil(len(files) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, kind, image in zip(axs.flat, kinds, images):
        ax.imshow(image.T, cmap="gray", origin="lower", vmin=0, vmax=255)
        ax.set_title(f"{kind}_{RESOLUTION}_0")
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0, hspace=0.12, wspace=0.02)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for (source, kind, i), image in zip(BOOK_SAMPLES, images):
        binary = 255 * (image / 255 >= THRESHOLD)
        for suffix, field in (("", image), ("_binary", binary)):
            fig = plt.figure(figsize=(1, 1))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(field.T, cmap="gray", origin="lower", vmin=0, vmax=255, interpolation="none")
            ax.axis("off")
            fig.savefig(RGB_PDF_DIR / f"benchmark_{source}_{kind.lower()}{suffix}.pdf")
            plt.close(fig)
