"""geometry of the PINN geometry study (helper.gyroid_mask) at the voxel centres, clamped
left, loaded downwards on the right. Red = solid not connected to the clamp (floating)."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from helper import gyroid_mask

R = 512
CELLS = [1, 1.5, 2, 3, 4, 6, 8, 12]  # continuous complexity, unit cells per side


def gyroid(cells):
    """solid mask (R, R) at the voxel centres."""
    axis = (np.arange(R) + 0.5) / R
    return gyroid_mask(np.stack(np.meshgrid(axis, axis, indexing="ij"), -1), cells)


fig, axes = plt.subplots(2, 4, figsize=(13, 7))
for ax, c in zip(axes.flat, CELLS):
    solid = gyroid(c)
    labels, _ = ndimage.label(solid)
    clamped = np.isin(labels, np.unique(labels[0][labels[0] > 0]))
    image = np.ones((R, R, 3))
    image[clamped] = 0.15
    image[solid & ~clamped] = [0.85, 0.1, 0.1]
    ax.imshow(image.transpose(1, 0, 2), origin="lower", extent=(0, 1, 0, 1))
    ax.plot([0, 0], [0, 1], color="tab:blue", lw=6)  # clamp
    for yy in np.linspace(0.2, 0.95, 5):
        ax.annotate("", (1.04, yy - 0.12), (1.04, yy), arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_xlim(-0.03, 1.1)
    ax.set_axis_off()
    loaded = clamped[-1].mean()
    ax.set_title(f"{c} cells/side, solid {solid.mean():.0%}\n"
                 f"floating {(solid & ~clamped).sum() / solid.sum():.1%}, "
                 f"right edge solid {loaded:.0%}", fontsize=9)
fig.tight_layout()
plt.show()
