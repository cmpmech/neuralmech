"""Generate two paired datasets of synthetic fiber images.

The paired dataset has one circle per sample replaced by a square at the same location
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- configurable params -------------------------
SEED = 33
N = 256                # image size (N x N)
SAMPLES = 500           # number of samples to generate
MIN_CIRCLES = 2
MAX_CIRCLES = 6
RADIUS_RANGE = (0.065, 0.11)  # (min, max)
DOMAIN_LENGTH = 1.0

OUT_DIR = Path(__file__).parent / "../../data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_grid(N, domain_length=1.0):
    xs = np.linspace(0, domain_length, N)
    ys = np.linspace(0, domain_length, N)
    return np.meshgrid(xs, ys, indexing="ij")


def place_nonoverlapping(num, r, domain_length):
    """Return list of (x,y) centers for `num` objects with radii `rads`.
    Ensures center-to-center distance >= sum of radii.
    """
    centers = []
    tries = 0
    while len(centers) < num:
        xc = np.random.uniform(r, domain_length - r)
        yc = np.random.uniform(r, domain_length - r)

        ok = True
        for (xj, yj) in centers:
            if np.hypot(xc - xj, yc - yj) < (r + r):
                ok = False
                break
        if ok:
            centers.append((xc, yc))

        tries += 1
        if tries > 2000:
            raise RuntimeError("Could not place non-overlapping shapes; try smaller radii or fewer objects")

    return centers


def draw_circles(N, centers, rads, domain_length=1.0):
    x, y = make_grid(N, domain_length)
    img = np.zeros((N, N), dtype=np.uint8)
    for (xc, yc), r in zip(centers, rads):
        mask = (x - xc) ** 2 + (y - yc) ** 2 < r * r
        img[mask] = 1
    return img


def draw_square_at_center(N, center, half_width, domain_length=1.0):
    x, y = make_grid(N, domain_length)
    xc, yc = center
    mask = (np.abs(x - xc) <= half_width) & (np.abs(y - yc) <= half_width)
    img = np.zeros((N, N), dtype=np.uint8)
    img[mask] = 1
    return img


def main():
    np.random.seed(SEED)

    clean = np.zeros((SAMPLES, N, N), dtype=np.uint8)
    anom = np.zeros((SAMPLES, N, N), dtype=np.uint8)

    for i in range(SAMPLES):
        num_circles = np.random.randint(MIN_CIRCLES, MAX_CIRCLES + 1)
        # use a single radius for all circles in this sample (varies between samples)
        radius = float(np.random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1]))
        rads = np.full(num_circles, radius)

        centers = place_nonoverlapping(num_circles, radius, DOMAIN_LENGTH)

        # draw clean circles image
        img_clean = draw_circles(N, centers, rads, DOMAIN_LENGTH)

        # pick one index to replace by a square (deterministic per-seed)
        anom_idx = np.random.randint(0, num_circles)

        img_anom = img_clean.copy()
        # remove the circle at anom_idx, then draw square at same center
        (xc, yc) = centers[anom_idx]

        # erase the circle region
        xg, yg = make_grid(N, DOMAIN_LENGTH)
        circle_mask = (xg - xc) ** 2 + (yg - yc) ** 2 < radius * radius
        img_anom[circle_mask] = 0

        half_w = radius  # square with side 2*r centered at same point
        square_img = draw_square_at_center(N, (xc, yc), half_w, DOMAIN_LENGTH)
        img_anom[square_img.astype(bool)] = 1

        clean[i] = img_clean
        anom[i] = img_anom

    # save paired datasets (same sample order)
    f_clean = OUT_DIR / f"fibers_{N}_{SAMPLES}.npy"
    f_anom = OUT_DIR / f"fibers_{N}_{SAMPLES}_1anom.npy"
    np.save(f_clean, clean)
    np.save(f_anom, anom)
    print(f"saved clean dataset to: {f_clean}")
    print(f"saved 1-anom dataset to: {f_anom}")

    # save a quick preview of first sample
    fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=128)
    ax[0].imshow(clean[0], cmap="binary", vmin=0, vmax=1)
    ax[0].set_title("clean")
    ax[1].imshow(anom[0], cmap="binary", vmin=0, vmax=1)
    ax[1].set_title("1-anom (square)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    preview = Path(__file__).parent / "../../results/fibers_1anom_preview.png"
    fig.savefig(preview, bbox_inches="tight", pad_inches=0)
    print(f"saved preview to: {preview}")


if __name__ == "__main__":
    main()
