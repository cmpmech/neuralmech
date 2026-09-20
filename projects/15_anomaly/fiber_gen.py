import matplotlib.pyplot as plt
import numpy as np

np.random.seed(45)


# -------------------------------- helper --------------------------------
def generate_circles(N, num_circles, radius=0.1, domain_length=1, iters=2000):
    if num_circles * np.pi * radius**2 > 0.7 * domain_length**2:
        return None  # relaxation never converges above ~0.7, skip the wasted iterations
    spacing = 2 * radius + 2 * domain_length / N  # one pixel gap between fibers
    centers = np.random.uniform(radius, domain_length - radius, (num_circles, 2))
    for it in range(iters):
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

    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing="ij")
    domain = np.zeros((N, N))
    for xc, yc in centers:
        domain[(x - xc) ** 2 + (y - yc) ** 2 < radius**2] = 1
    return domain


def generate_squares(
    N, num_circles, num_squares, half_width=0.1, domain_length=1, max_attempts=200
):
    domain = generate_circles(N, num_circles, half_width, domain_length)
    if domain is None:
        return None
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing="ij")

    width = 2 * half_width
    for i in range(num_squares):
        for attempt in range(max_attempts):
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


# ------------------------------- settings -------------------------------
N = 256
# N = 128
MAX_CIRCLES = 20
MAX_RADIUS = 0.2

# ------------------------- generate normal data -------------------------
samples = 500

domains = np.zeros((samples, N, N))
for i in range(samples):
    domain = None
    while domain is None:  # redraw count and radius when the packing is infeasible
        num_circles = np.random.randint(5, MAX_CIRCLES + 1)
        radius = np.random.uniform(0.05, MAX_RADIUS)
        domain = generate_circles(N, num_circles, radius)
    domains[i] = domain

volume_fraction = domains.mean(axis=(1, 2))
print(
    f"volume fraction span [{volume_fraction.min():.3f}, {volume_fraction.max():.3f}], "
    f"mean {volume_fraction.mean():.3f}"
)
counts, edges = np.histogram(volume_fraction, bins=10)
for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
    print(f"  [{lo:.2f}, {hi:.2f}): {count}")

# -------------------------------- export --------------------------------
np.save(f"../../data/fibers_{N}.npy", domains)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
ax.imshow(domains[0], cmap="binary")
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"../../results/rgb_pdf/fibers.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


# -------------------------- generate anomalies --------------------------
samples = 50
num_circles = 10


for num_squares in range(num_circles + 1):
    domains = np.zeros((samples, N, N))
    for i in range(samples):
        radius = np.random.uniform(0.05, 0.1)
        radius = 0.08
        domain = None
        while domain is None:
            domain = generate_squares(N, num_circles - num_squares, num_squares, radius)
        domains[i] = domain

# -------------------------------- export --------------------------------
    np.save(f"../../data/fibers_anomaly_{num_squares}_{N}.npy", domains)

# ---------------------------- postprocessing ----------------------------
    if num_squares == 1:
        fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
        ax.imshow(domains[0], cmap="binary")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            f"../../results/rgb_pdf/fibers_anomaly.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()

# -------------------------- structured fibers ---------------------------
fibers = 4
domain = np.zeros((N, N))
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y, indexing="ij")
dx = 1 / (4 + (fibers - 1) * 3)
for i in range(fibers):
    for j in range(fibers):
        xc, yc = (2 + 3 * i) * dx, (2 + 3 * j) * dx
        mask = (x - xc) ** 2 + (y - yc) ** 2 < (dx * 0.9) ** 2
        domain[mask] = 1

# -------------------------------- export --------------------------------
np.save(f"../../data/fibers_anomaly_structured_{N}.npy", domain)
