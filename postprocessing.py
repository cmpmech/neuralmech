import datetime
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    LogNorm,
    SymLogNorm,
)


# --------------------------- exporting to csv ---------------------------
def save_csv(path: str, **cols) -> None:
    """Save keyword-argument columns as a space-separated CSV file."""
    pd.DataFrame(cols).to_csv(path, sep=" ", index=False)


# ------------------------ temporary figure saving -----------------------
def save_temp_fig(name: str) -> None:
    """Save the current figure to '<name>_<timestamp>.jpg' for quick inspection.

    A throwaway replacement for plt.savefig during development: pass a path prefix
    that already includes the results folder (e.g. 'results/analog_rnn'); a
    timestamp is appended so repeated calls never overwrite each other.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{name}_{stamp}.jpg")


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


# ------------------------- image postprocessing -------------------------
def show_image(
    img: np.ndarray,
    grayscale: bool = False,
    path: str | None = None,
    close: bool = False,
) -> None:
    """Display a single image without axes.

    Args:
        img:       Numpy array of shape (H, W) for grayscale or (H, W, 3) for RGB.
                   Values are shown as-is: normalise before calling if needed.
        grayscale: If True, render with a gray colormap.
        path:      Optional file path to save the figure (e.g. 'results/out.pdf').
    """
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, cmap="gray" if grayscale else None)
    ax.axis("off")
    ax.set_rasterized(True)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if path is not None:
        plt.savefig(path)
    plt.show() if not close else plt.close()


def show_colorbar(
    cb,
    path: str | None = None,
    close: bool = False,
    orientation: str = "vertical",
    aspect_ratio: float = 16,
) -> None:
    """Display a standalone, tick-free colorbar for a mappable.

    Args:
        cb:           A matplotlib mappable (e.g. the return of ax.contourf or
                      ax.pcolormesh) whose colormap and normalisation are drawn.
        path:         Optional file path to save the figure (e.g. 'results/bar.pdf').
        close:        If True, close the figure instead of showing it interactively.
        orientation:  'vertical' (default) or 'horizontal' colorbar layout.
        aspect_ratio: Length of the bar relative to its 0.5-inch thickness.
    """
    norm = cb.norm
    if isinstance(norm, LogNorm):
        scale = "log"
    elif isinstance(norm, SymLogNorm):
        scale = "symlog"
    else:
        scale = "linear"
    print(f"colorbar: min={norm.vmin:.2e}, max={norm.vmax:.2e}, scale={scale}")

    if orientation == "horizontal":
        figsize = (0.5 * aspect_ratio, 0.5)
    else:
        figsize = (0.5, 0.5 * aspect_ratio)

    fig = plt.figure(figsize=figsize, dpi=100)
    cax = fig.add_axes([0, 0, 1, 1])
    mappable = ScalarMappable(norm=norm, cmap=cb.cmap)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.ax.tick_params(length=0)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if path is not None:
        plt.savefig(path)
    plt.show() if not close else plt.close()


def cmyk_to_rgb(c, m, y, k):
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)
    return (r, g, b)
