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
        overlap = True
        while overlap == True:
            xc = np.random.uniform(radius, domain_length - radius)
            yc = np.random.uniform(radius, domain_length - radius)

            mask = (x - xc) ** 2 + (y - yc) ** 2 < radius**2
            mask_half = (x - xc) ** 2 + (y - yc) ** 2 < (radius / 2) ** 2

            if ~np.any(domain[mask] == 1):
                overlap = False
        domain[mask] = 1
        domain[mask_half] = 0.5

    return domain


# ------------------------------- settings -------------------------------
N = 128
MASK_RATIO = 0.6

# ------------------------- generate normal data -------------------------
samples = 128  # 256 #128 #8 #256 #64 #64 #64  # 32 #10 #200  # 500

for j in range(2):
    domains = np.zeros((samples, N, N))
    for i in range(samples):
        num_circles = np.random.randint(5, 10)
        radius = np.random.uniform(0.05, 0.1)
        domains[i] = generate_circles(N, num_circles, radius)

# -------------------------------- export --------------------------------
    if j == 0:
        np.save(f"../../data/graded_fibers_{N}.npy", domains)
    else:
        np.save(f"../../data/graded_fibers_test_{N}.npy", domains)
        n_observed = round((1 - MASK_RATIO) * N * N)
        flat = np.zeros(N * N, dtype=bool)
        flat[:n_observed] = True
        masks = np.array([np.random.permutation(flat).reshape(N, N) for _ in range(samples)])
        np.save(f"../../data/graded_fiber_masks_test_{N}.npy", masks)
