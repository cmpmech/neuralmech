import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solvers.optimization import DensityFilter, projection

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# design
RESOLUTION = 300
RADII = [10, 18, 26, 36]  # 0.33, 0.6, 0.87 and 1.2 times the filter radius

# reparametrization
FILTER_RADIUS = 30
BETA = 8.0  # projection sharpness
ETA = 0.5  # projection threshold

# ------------------------------------ create data ------------------------------------
quarter = RESOLUTION // 4
centers = [(quarter, 3 * quarter), (3 * quarter, 3 * quarter),
           (quarter, quarter), (3 * quarter, quarter)]  # reading order, smallest first

x, y = np.meshgrid(np.arange(RESOLUTION), np.arange(RESOLUTION), indexing="ij")
design = np.zeros((RESOLUTION, RESOLUTION))
for (center_x, center_y), radius in zip(centers, RADII):
    design[(x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2] = 1.0

# --------------------------------- reparametrization ---------------------------------
density_filter = DensityFilter(FILTER_RADIUS, design.shape)
filtered = density_filter(design)
projected = projection(filtered, BETA, ETA)

# ----------------------------------- postprocessing ----------------------------------
for radius, center in zip(RADII, centers):
    print(f"radius {radius:3d}  filtered {filtered[center]:.3f} "
          f" projected {projected[center]:.3f}")

fields = [design, filtered, projected]
names = ["reparametrization_design", "reparametrization_filtered",
         "reparametrization_projected"]

if not args.book:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, field in zip(axes, fields):
        ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, field in zip(names, fields):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf")
        plt.close(fig)
