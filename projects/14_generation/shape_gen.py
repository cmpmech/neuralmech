import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate


def generate_circle(N, domain_length=1, radius=0.1):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)

    xc = np.random.uniform(radius, domain_length - radius)
    yc = np.random.uniform(radius, domain_length - radius)

    mask = (x - xc) ** 2 + (y - yc) ** 2 < radius**2

    domain[mask] = 1
    return domain


def generate_square(N, domain_length=1, w=0.2):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(w / 2, domain_length - w / 2)
    yc = np.random.uniform(w / 2, domain_length - w / 2)
    angle = np.random.uniform(0, 2 * np.pi)
    # rotate query points around center (inverse rotation)
    dx, dy = x - xc, y - yc
    xr = dx * np.cos(angle) + dy * np.sin(angle)
    yr = -dx * np.sin(angle) + dy * np.cos(angle)
    domain[(np.abs(xr) < w / 2) & (np.abs(yr) < w / 2)] = 1
    return domain


def generate_triangle(N, domain_length=1, size=0.2):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(size, domain_length - size)
    yc = np.random.uniform(size, domain_length - size)
    angle = np.random.uniform(0, 2 * np.pi)
    dx, dy = x - xc, y - yc
    xr = dx * np.cos(angle) + dy * np.sin(angle)
    yr = -dx * np.sin(angle) + dy * np.cos(angle)
    # fixed upright equilateral triangle in rotated frame
    vx = size * np.cos(np.array([np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]))
    vy = size * np.sin(np.array([np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]))
    d1 = (xr - vx[1]) * (vy[0] - vy[1]) - (vx[0] - vx[1]) * (yr - vy[1])
    d2 = (xr - vx[2]) * (vy[1] - vy[2]) - (vx[1] - vx[2]) * (yr - vy[2])
    d3 = (xr - vx[0]) * (vy[2] - vy[0]) - (vx[2] - vx[0]) * (yr - vy[0])
    domain[
        ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    ] = 1
    return domain


def generate_ellipse(N, domain_length=1, a=0.25, b=0.1):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(a, domain_length - a)
    yc = np.random.uniform(b, domain_length - b)
    angle = np.random.uniform(0, 2 * np.pi)
    dx, dy = x - xc, y - yc
    xr = dx * np.cos(angle) + dy * np.sin(angle)
    yr = -dx * np.sin(angle) + dy * np.cos(angle)
    domain[(xr / a) ** 2 + (yr / b) ** 2 < 1] = 1
    return domain


def generate_star(N, domain_length=1, r_outer=0.15, r_inner=0.06, n_points=5):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(r_outer, domain_length - r_outer)
    yc = np.random.uniform(r_outer, domain_length - r_outer)
    angle = np.random.uniform(0, 2 * np.pi)
    dx, dy = x - xc, y - yc
    xr = dx * np.cos(angle) + dy * np.sin(angle)
    yr = -dx * np.sin(angle) + dy * np.cos(angle)

    # alternating outer/inner vertices
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n_points, endpoint=False)
    radii = np.tile([r_outer, r_inner], n_points)
    vx = radii * np.cos(angles)
    vy = radii * np.sin(angles)

    # ray casting point-in-polygon
    inside = np.zeros(xr.shape, dtype=bool)
    n = len(vx)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vx[i], vy[i]
        xj, yj = vx[j], vy[j]
        intersect = ((yi > yr) != (yj > yr)) & (
            xr < (xj - xi) * (yr - yi) / (yj - yi) + xi
        )
        inside ^= intersect

    domain[inside] = 1
    return domain


N = 128  # 256
samples = 512  # 128  # per shape
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y, indexing="ij")

generators = [
    generate_circle,
    generate_square,
    generate_triangle,
    generate_ellipse,
    generate_star,
]
labels = ["circle", "square", "triangle", "ellipse", "star"]
for label, generator in zip(labels, generators):
    domains = np.zeros((samples, N, N))
    for sample in range(samples):
        domains[sample] = generator(N)

    np.save(f"../../data/shapes_{label}_{N}.npy", domains.astype(np.float32))

    fig, ax = plt.subplots(figsize=(2, 2), dpi=N)
    ax.pcolormesh(x, y, domains[0], cmap="binary")  # imshow
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_rasterized(True)
    fig.tight_layout(pad=0)
    plt.savefig(f"../../results/shapes_{label}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close()
#    plt.show()
