import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from postprocessing import load_cmap, show_colorbar

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
ASPECT = 16

rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

COLORMAPS = [
    ("cividis", "cividis"),
    ("Spectral", "Spectral"),
    ("binary", "binary"),
    ("hot_r", "hot_r"),
    ("turbo", "turbo"),
    ("rainbow", rainbow),
    ("cmr_torch", cmr.torch),
    ("cmr_fusion", cmr.fusion),
    ("cmr_guppy", cmr.guppy),  # instead of pride -> for fluids
    ("cmr_infinity", cmr.infinity),
]

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    grad = np.linspace(0, 1, 256)
    n = len(COLORMAPS)
    fig, axes = plt.subplots(n, 1, figsize=(5, n * 0.6))
    for i, (name, cmap) in enumerate(COLORMAPS):
        axes[i].imshow(grad.reshape(1, -1), aspect="auto", cmap=cmap)
        axes[i].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    norm = Normalize(0, 1)
    for name, cmap in COLORMAPS:
        for orientation in ["vertical", "horizontal"]:
            suffix = "v" if orientation == "vertical" else "h"
            show_colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                path=RGB_PDF_DIR / f"colorbar_{name}_{suffix}.pdf",
                close=True,
                orientation=orientation,
                aspect_ratio=ASPECT,
            )
