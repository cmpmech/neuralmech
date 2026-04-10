import matplotlib.pyplot as plt
import numpy as np


def generate_circle(N, domain_length=1, radius=0.4):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)

    xc = np.random.uniform(radius, domain_length - radius)
    yc = np.random.uniform(radius, domain_length - radius)

    mask = (x - xc) ** 2 + (y - yc) ** 2 < radius**2

    domain[mask] = 1
    return domain


def generate_square(N, domain_length=1, w=0.8):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(w / 2, domain_length - w / 2)
    yc = np.random.uniform(w / 2, domain_length - w / 2)
    dx, dy = x - xc, y - yc
    mask = (np.abs(dx) < w / 2) & (np.abs(dy) < w / 2)

    domain[mask] = 1
    return domain


def generate_triangle(N, domain_length=1, size=0.45):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(size, domain_length - size)
    yc = np.random.uniform(size, domain_length - size)
    dx, dy = x - xc, y - yc
    # fixed upright equilateral triangle
    vx = size * np.cos(np.array([np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]))
    vy = size * np.sin(np.array([np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]))
    d1 = (dx - vx[1]) * (vy[0] - vy[1]) - (vx[0] - vx[1]) * (dy - vy[1])
    d2 = (dx - vx[2]) * (vy[1] - vy[2]) - (vx[1] - vx[2]) * (dy - vy[2])
    d3 = (dx - vx[0]) * (vy[2] - vy[0]) - (vx[2] - vx[0]) * (dy - vy[0])
    mask = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))

    domain[mask] = 1
    return domain


def generate_ellipse(N, domain_length=1, a=0.4, b=0.1):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(a, domain_length - a)
    yc = np.random.uniform(b, domain_length - b)
    dx, dy = x - xc, y - yc
    mask = (dx / a) ** 2 + (dy / b) ** 2 < 1

    domain[mask] = 1
    return domain


def generate_star(N, domain_length=1, r_outer=0.4, r_inner=0.2, n_points=5):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(r_outer, domain_length - r_outer)
    yc = np.random.uniform(r_outer, domain_length - r_outer)
    dx, dy = x - xc, y - yc

    # alternating outer/inner vertices
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n_points, endpoint=False)
    radii = np.tile([r_outer, r_inner], n_points)
    vx = radii * np.cos(angles)
    vy = radii * np.sin(angles)

    # ray casting point-in-polygon
    mask = np.zeros(dx.shape, dtype=bool)
    n = len(vx)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vx[i], vy[i]
        xj, yj = vx[j], vy[j]
        intersect = ((yi > dy) != (yj > dy)) & (
            dx < (xj - xi) * (dy - yi) / (yj - yi) + xi
        )
        mask ^= intersect

    domain[mask] = 1
    return domain


def generate_cross(N, domain_length=1, w=0.2, h=0.8):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)
    xc = np.random.uniform(h / 2, domain_length - h / 2)
    yc = np.random.uniform(h / 2, domain_length - h / 2)
    dx, dy = x - xc, y - yc
    horizontal = (np.abs(dx) < h / 2) & (np.abs(dy) < w / 2)
    vertical = (np.abs(dx) < w / 2) & (np.abs(dy) < h / 2)
    mask = horizontal | vertical

    domain[mask] = 1
    return domain


if __name__ == "__main__":
    N = 128
    samples = 128  # per shape

    generators = [
        generate_circle,
        generate_square,
        generate_triangle,
        generate_ellipse,
        generate_star,
        generate_cross,
    ]
    labels = ["circle", "square", "triangle", "ellipse", "star", "cross"]

# --------------------------- data generation ----------------------------
    for label, generator in zip(labels, generators):
        domains = np.zeros((samples, N, N))
        for sample in range(samples):
            domains[sample] = generator(N)

        np.save(f"../../data/shapes_{label}_{N}.npy", domains.astype(np.float32))

# ------------------------- book postprocessing --------------------------
        fig, ax = plt.subplots(figsize=(2, 2), dpi=N)
        ax.imshow(domains[0].T, cmap="binary", origin='lower')
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.tight_layout(pad=0)
        plt.savefig(
            f"../../results/shapes_{label}.pdf", bbox_inches="tight", pad_inches=0
        )
        plt.close()
