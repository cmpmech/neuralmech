import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------- exporting to csv ---------------------------
def save_csv(path: str, **cols) -> None:
    """Save keyword-argument columns as a space-separated CSV file."""
    pd.DataFrame(cols).to_csv(path, sep=" ", index=False)


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
