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
# design: a unit plate with a circular hole, once with a wide and once with a narrow
# transition between void and material
RESOLUTION = 800
RADIUS = 0.25
SMOOTH_WIDTH = 0.05  # transition half-width of the smooth hole
SHARP_WIDTH = 0.005  # ten times narrower, four cells wide

# penalties
EPSILON = 1e-3  # total variation smoothing

# ------------------------------------ create data ------------------------------------
h = 1.0 / RESOLUTION
coordinates = (np.arange(RESOLUTION) + 0.5) * h
x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

# tanh profile across the hole boundary: 0 inside the hole, 1 in the material
designs = [0.5 * (1.0 + np.tanh((radius - RADIUS) / width))
           for width in [SMOOTH_WIDTH, SHARP_WIDTH]]

# ------------------------------------- penalties -------------------------------------
def gradient_norm(design):
    """cell-wise |grad gamma| by central differences, one-sided at the boundary."""
    dx, dy = np.gradient(design, h, edge_order=2)
    return np.sqrt(dx**2 + dy**2)


def tikhonov(design):  # 1/2 int |grad gamma|^2 dOmega
    return 0.5 * np.sum(gradient_norm(design) ** 2) * h**2


def total_variation(design):  # int sqrt(|grad gamma|^2 + eps^2) dOmega
    return np.sum(np.sqrt(gradient_norm(design) ** 2 + EPSILON**2)) * h**2


# ----------------------------------- postprocessing ----------------------------------
names = ["penalization_smooth", "penalization_sharp"]
widths = [SMOOTH_WIDTH, SHARP_WIDTH]

for name, width, design in zip(names, widths, designs):
    # the analytical values for a tanh profile of half-width w along a circle of
    # circumference 2 pi R: pi R / (3 w) and 2 pi R (the smoothing adds eps |Omega|)
    print(f"{name.split('_')[1]:6s}  width {width:.3f}  "
          f"tikhonov {tikhonov(design):7.3f} (exact {np.pi * RADIUS / (3 * width):7.3f})  "
          f"total variation {total_variation(design):6.3f} "
          f"(exact {2 * np.pi * RADIUS + EPSILON:6.3f})")

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, name, design in zip(axes, names, designs):
        ax.imshow(design.T, origin="lower", cmap="binary_r", vmin=0.0, vmax=1.0)
        ax.set_title(name.split("_")[1])
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, design in zip(names, designs):
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.imshow(design.T, origin="lower", cmap="binary_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf")
        plt.close(fig)
