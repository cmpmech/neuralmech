import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

np.random.seed(45)

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 256
MAX_CIRCLES = 20
MAX_RADIUS = 0.2

NORMAL_SAMPLES = 1000
ANOMALY_SAMPLES = 50
ANOMALY_CIRCLES = 10
ANOMALY_RADIUS = 0.08
STRUCTURED_FIBERS = 4


# --------------------------------------- helper --------------------------------------
def generate_circles(resolution, circles, radius=0.1, domain_length=1, iters=2000):
    if circles * np.pi * radius**2 > 0.7 * domain_length**2:
        return None  # relaxation never converges above ~0.7, skip the wasted iterations
    spacing = 2 * radius + 2 * domain_length / resolution  # one pixel gap between fibers
    centers = np.random.uniform(radius, domain_length - radius, (circles, 2))
    for _ in range(iters):
        diff = centers[:, None] - centers[None]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        overlap = np.clip(spacing - dist, 0, None)
        if overlap.max(initial=0) < 1e-6:
            break
        push = overlap[..., None] * diff / np.maximum(dist, 1e-9)[..., None]
        centers += 0.5 * push.sum(axis=1)
        centers = np.clip(centers, radius, domain_length - radius)
    else:
        return None  # packing stalled (domain too crowded), caller redraws

    x = np.linspace(0, domain_length, resolution)
    y = np.linspace(0, domain_length, resolution)
    x, y = np.meshgrid(x, y, indexing="ij")
    domain = np.zeros((resolution, resolution))
    for xc, yc in centers:
        domain[(x - xc) ** 2 + (y - yc) ** 2 < radius**2] = 1
    return domain


def generate_squares(
    resolution, circles, squares, half_width=0.1, domain_length=1, max_attempts=200
):
    domain = generate_circles(resolution, circles, half_width, domain_length)
    if domain is None:
        return None
    x = np.linspace(0, domain_length, resolution)
    y = np.linspace(0, domain_length, resolution)
    x, y = np.meshgrid(x, y, indexing="ij")

    width = 2 * half_width
    for _ in range(squares):
        for _ in range(max_attempts):
            x0 = np.random.uniform(0, domain_length - width)
            y0 = np.random.uniform(0, domain_length - width)

            mask = (x - x0) > 0
            mask *= (x - (x0 + width)) < 0
            mask *= (y - y0) > 0
            mask *= (y - (y0 + width)) < 0
            if not np.any(domain[mask] == 1):
                domain[mask] = 1
                break
        else:
            return None
    return domain


# ------------------------------------ create data ------------------------------------
domains = np.zeros((NORMAL_SAMPLES, RESOLUTION, RESOLUTION))
for i in range(NORMAL_SAMPLES):
    domain = None
    while domain is None:  # redraw count and radius when the packing is infeasible
        circles = np.random.randint(5, MAX_CIRCLES + 1)
        radius = np.random.uniform(0.05, MAX_RADIUS)
        domain = generate_circles(RESOLUTION, circles, radius)
    domains[i] = domain

volume_fraction = domains.mean(axis=(1, 2))
print(
    f"volume fraction span [{volume_fraction.min():.3f}, {volume_fraction.max():.3f}], "
    f"mean {volume_fraction.mean():.3f}"
)
counts, edges = np.histogram(volume_fraction, bins=10)
for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
    print(f"  [{lo:.2f}, {hi:.2f}): {count}")

np.save(DATA_DIR / f"fibers_{RESOLUTION}.npy", domains)

normal_sample = domains[0]

# ---------------------------------- create anomalies ---------------------------------
anomaly_sample = None
for squares in range(ANOMALY_CIRCLES + 1):
    domains = np.zeros((ANOMALY_SAMPLES, RESOLUTION, RESOLUTION))
    for i in range(ANOMALY_SAMPLES):
        domain = None
        while domain is None:
            domain = generate_squares(
                RESOLUTION, ANOMALY_CIRCLES - squares, squares, ANOMALY_RADIUS
            )
        domains[i] = domain

    np.save(DATA_DIR / f"fibers_anomaly_{squares}_{RESOLUTION}.npy", domains)
    if squares == 1:
        anomaly_sample = domains[0]

# --------------------------------- structured fibers ---------------------------------
domain = np.zeros((RESOLUTION, RESOLUTION))
x = np.linspace(0, 1, RESOLUTION)
y = np.linspace(0, 1, RESOLUTION)
x, y = np.meshgrid(x, y, indexing="ij")
dx = 1 / (4 + (STRUCTURED_FIBERS - 1) * 3)
for i in range(STRUCTURED_FIBERS):
    for j in range(STRUCTURED_FIBERS):
        xc, yc = (2 + 3 * i) * dx, (2 + 3 * j) * dx
        domain[(x - xc) ** 2 + (y - yc) ** 2 < (dx * 0.9) ** 2] = 1

np.save(DATA_DIR / f"fibers_anomaly_structured_{RESOLUTION}.npy", domain)

# ----------------------------------- postprocessing ----------------------------------
figures = {"fibers.pdf": normal_sample, "fibers_anomaly.pdf": anomaly_sample}
for name, sample in figures.items():
    fig, ax = plt.subplots(figsize=(1, 1), dpi=RESOLUTION)
    ax.imshow(sample, cmap="binary")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if not args.book:
        plt.show()
# -------------------------------- book postprocessing --------------------------------
    else:
        plt.savefig(RGB_PDF_DIR / name)
        plt.close(fig)
