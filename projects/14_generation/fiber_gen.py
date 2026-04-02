import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# HELPER

def generate_circles(N, num_circles, radius=0.1, domain_length=1):
    domain = np.ones((N, N))
    x = np.linspace(0, domain_length, N)
    y = np.linspace(0, domain_length, N)
    x, y = np.meshgrid(x, y)

    for i in range(num_circles):
        overlap = True
        while overlap == True:
            xc = np.random.uniform(radius, domain_length - radius)
            yc = np.random.uniform(radius, domain_length - radius)

            mask = (x - xc)**2 + (y - yc)**2 < radius**2
            if ~np.any(domain[mask] == -1):
                overlap = False
        domain[mask] = -1
    return domain

# MICROSTRUCTURE GENERATION
samples = 200
N = 128 #256

domains = np.zeros((samples, N, N))
for i in range(samples):
    num_circles = np.random.randint(1, 10)
    radius = np.random.uniform(0.02, 0.1)
    domains[i] = generate_circles(N, num_circles, radius)

# ----------------------------- export data ------------------------------
np.save(f'../../data/fibers_{N}.npy', domains)

# POSTPROCESSING
fig, ax = plt.subplots(figsize=(2,2), dpi=samples)
ax.imshow(domains[0], cmap='binary_r')
ax.set_aspect('equal')
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.show()

