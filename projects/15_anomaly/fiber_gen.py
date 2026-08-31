import matplotlib.pyplot as plt
import numpy as np

np.random.seed(45)


# -------------------------------- helper --------------------------------
def generate_circles(N, num_circles, radius=0.1, domain_length=1):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing="ij")

    for i in range(num_circles):
        # TODO enable to give every fiber its own radius. the caller draws one radius
        # per image, so today every fiber in an image is the same size; drawing it here
        # instead varies the size within the image. two things follow: the anomaly sets
        # below get varying circles too (their squares keep half_width), and a large
        # radius drawn into an already crowded domain can leave the placement loop
        # spinning, so cap the count or shrink the range if it stalls
        # radius = np.random.uniform(0.05, 0.1)
        overlap = True
        while overlap == True:
            xc = np.random.uniform(radius, domain_length - radius)
            yc = np.random.uniform(radius, domain_length - radius)

            mask = (x - xc) ** 2 + (y - yc) ** 2 < radius**2
            if ~np.any(domain[mask] == 1):
                overlap = False
        domain[mask] = 1
    return domain


def generate_squares(N, num_circles, num_squares, half_width=0.1, domain_length=1):
    domain = generate_circles(N, num_circles, half_width, domain_length)
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing="ij")

    width = 2 * half_width
    for i in range(num_squares):
        overlap = True
        while overlap == True:
            x0 = np.random.uniform(0, domain_length - width)
            y0 = np.random.uniform(0, domain_length - width)

            mask = (x - x0) > 0
            mask *= (x - (x0 + width)) < 0
            mask *= (y - y0) > 0
            mask *= (y - (y0 + width)) < 0
            if ~np.any(domain[mask] == 1):
                overlap = False
        domain[mask] = 1
    return domain


# ------------------------------- settings -------------------------------
N = 256
# N = 128

# ------------------------- generate normal data -------------------------
samples = 500

domains = np.zeros((samples, N, N))
for i in range(samples):
    num_circles = np.random.randint(5, 10)
    radius = np.random.uniform(0.05, 0.1)
    domains[i] = generate_circles(N, num_circles, radius)

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
        # radius = np.random.uniform(0.02, 0.2)
        radius = np.random.uniform(0.05, 0.1)
        radius = 0.08
        domains[i] = generate_squares(N, num_circles - num_squares, num_squares, radius)

# -------------------------------- export --------------------------------
    np.save(f"../../data/fibers_anomaly_{num_squares}_{N}.npy", domains)

# ---------------------------- postprocessing ----------------------------
    if num_squares == 1:
        fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
        ax.imshow(domains[0], cmap="binary")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            f"../../results/rgb_pdf/fibers_anomaly.pdf", bbox_inches="tight", pad_inches=0
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
