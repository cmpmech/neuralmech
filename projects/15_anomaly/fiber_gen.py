import matplotlib.pyplot as plt
import numpy as np

seed = 45


# -------------------------------- helper --------------------------------
def generate_circles(N, num_circles, radius=0.1, domain_length=1):
    domain = np.zeros((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y, indexing="ij")

    for i in range(num_circles):
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

np.random.seed(seed)
# ------------------------------- settings -------------------------------
N = 256     # image size N x N
min_circles, max_circles = 1, 3

# ------------------------- generate normal data -------------------------
samples = 400

domains = np.zeros((samples, N, N))
for i in range(samples):
    num_total = np.random.randint(min_circles+1, max_circles + 1)
    radius = np.random.uniform(0.065, 0.11)
    domains[i] = generate_circles(N, num_total, radius)

# -------------------------------- export --------------------------------
fpathstart = f"../../data/t"
fpath = f"{fpathstart}_circ{min_circles}min_{max_circles}max_{samples}.npy"
np.save(fpath, domains)
print(f"saved {samples} samples of fibers to {fpath}")

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
ax.imshow(domains[0], cmap="binary")
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"../../results/fibers.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


np.random.seed(seed)
# -------------------------- generate anomalies --------------------------
# samples = 50
# num_circles = 10
num_total=2

# for num_squares in range(num_circles + 1):
num_squares = 1
domains = np.zeros((samples, N, N))
for i in range(samples):
    # radius = np.random.uniform(0.02, 0.2)
    radius = np.random.uniform(0.04, 0.08)
    # radius = 0.08
    domains[i] = generate_squares(N, num_total - num_squares, num_squares, radius)

# -------------------------------- export --------------------------------
fpath = f"{fpathstart}_{num_squares}sq_{num_total}tot_{samples}.npy"
np.save(fpath, domains)

print(f"saved {samples} samples of fibers with {num_squares} squares to {fpath}")
# ---------------------------- postprocessing ----------------------------
    # if num_squares == 1:
fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
ax.imshow(domains[4], cmap="binary")
ax.axis("off")
plt.show()
