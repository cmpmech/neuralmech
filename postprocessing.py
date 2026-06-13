import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# --------------------------- exporting to csv ---------------------------
def save_csv(path: str, **cols) -> None:
    """Save keyword-argument columns as a space-separated CSV file."""
    pd.DataFrame(cols).to_csv(path, sep=" ", index=False)


# ---------------------------- colormaps ---------------------------------
def load_cmap(path: str) -> LinearSegmentedColormap:
    """Build a matplotlib colormap from a ParaView .cmap (JSON) file.

    The .cmap RGBPoints are stored as [x0, r0, g0, b0, x1, r1, g1, b1, ...];
    positions are rescaled to [0, 1] for matplotlib.
    """
    data = json.loads(Path(path).read_text())
    pts = data["RGBPoints"]
    xs, rs, gs, bs = pts[0::4], pts[1::4], pts[2::4], pts[3::4]
    lo, hi = xs[0], xs[-1]
    stops = [((x - lo) / (hi - lo), (r, g, b)) for x, r, g, b in zip(xs, rs, gs, bs)]
    return LinearSegmentedColormap.from_list(data.get("Name", "cmap"), stops)


# ---------------------------- image display -----------------------------
def show_image(
    img: np.ndarray,
    grayscale: bool = False,
    path: str | None = None,
    close: bool = False,
) -> None:
    """Display a single image without axes.

    Args:
        img:       Numpy array of shape (H, W) for grayscale or (H, W, 3) for RGB.
                   Values are shown as-is — normalise before calling if needed.
        grayscale: If True, render with a gray colormap.
        path:      Optional file path to save the figure (e.g. 'results/out.pdf').
    """
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, cmap="gray" if grayscale else None)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)

    if path is not None:
        plt.savefig(path)
    plt.show() if not close else plt.close()
