from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

np.random.seed(45)

# -------------------------------------- settings -------------------------------------
RESOLUTION = 128
MASK_RATIO = 0.6

SAMPLES = 128


# --------------------------------------- helper --------------------------------------
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


# ------------------------------------ create data ------------------------------------
for j in range(2):
    domains = np.zeros((SAMPLES, RESOLUTION, RESOLUTION))
    for i in range(SAMPLES):
        num_circles = np.random.randint(5, 10)
        radius = np.random.uniform(0.05, 0.1)
        domains[i] = generate_circles(RESOLUTION, num_circles, radius)

# --------------------------------------- export --------------------------------------
    if j == 0:
        np.save(DATA_DIR / f"graded_fibers_{RESOLUTION}.npy", domains)
    else:
        np.save(DATA_DIR / f"graded_fibers_test_{RESOLUTION}.npy", domains)
        n_observed = round((1 - MASK_RATIO) * RESOLUTION * RESOLUTION)
        flat = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
        flat[:n_observed] = True
        masks = np.array(
            [
                np.random.permutation(flat).reshape(RESOLUTION, RESOLUTION)
                for _ in range(SAMPLES)
            ]
        )
        np.save(DATA_DIR / f"graded_fiber_masks_test_{RESOLUTION}.npy", masks)
