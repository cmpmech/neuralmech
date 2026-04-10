import matplotlib.pyplot as plt
import numpy as np

np.random.seed(45)

# -------------------------------- helper --------------------------------
def generate_circles(N, num_circles, radius=0.1, domain_length=1):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing='ij')

    for i in range(num_circles):
        overlap = True
        while overlap == True:
            xc = np.random.uniform(radius, domain_length - radius)
            yc = np.random.uniform(radius, domain_length - radius)

            mask = (x - xc)**2 + (y - yc)**2 < radius**2
            if ~np.any(domain[mask] == 1):
                overlap = False
        domain[mask] = 1
    return domain

def generate_squares(N, num_circles, num_squares, width=0.2,
                     domain_length = 1):
    domain = generate_circles(N, num_circles, width / 2, domain_length)
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing='ij')

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
N = 256 #128

# ------------------------- generate normal data -------------------------
samples = 500

domains = np.zeros((samples, N, N))
for i in range(samples):
    num_circles = np.random.randint(1, 10)
    radius = np.random.uniform(0.02, 0.1)
    domains[i] = generate_circles(N, num_circles, radius)

# -------------------------------- export --------------------------------
np.save(f'../../data/fibers_{N}.npy', domains)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(2,2), dpi=N)
ax.imshow(domains[0], cmap='binary')
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    f"../../results/fibers.pdf", bbox_inches="tight", pad_inches=0
)
plt.show()


# -------------------------- generate anomalies --------------------------
samples = 50
num_circles = 10


for num_squares in range(num_circles):
    domains = np.zeros((samples, N, N))
    for i in range(samples):
        radius = np.random.uniform(0.02, 0.2)
        domains[i] = generate_squares(N, num_circles - num_squares,
                                      num_squares, radius)

# -------------------------------- export --------------------------------
    np.save(f'../../data/fibers_anomaly_{num_squares}_{N}.npy', domains)

# ---------------------------- postprocessing ----------------------------
    if num_squares == 1:

        fig, ax = plt.subplots(figsize=(2,2), dpi=N)
        ax.imshow(domains[0], cmap='binary')
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            f"../../results/fibers_anomaly.pdf", bbox_inches="tight",
            pad_inches=0
        )
        plt.show()
