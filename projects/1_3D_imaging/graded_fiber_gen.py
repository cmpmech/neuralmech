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
            # mask_quarter = (x - xc) ** 2 + (y - yc) ** 2 < (radius / 4) ** 2

            if ~np.any(domain[mask] == 1):
                overlap = False
        domain[mask] = 1
        domain[mask_half] = 0.5
        # domain[mask_quarter] = 0.25

    return domain


# ------------------------------- settings -------------------------------
# N = 256
N = 128

# ------------------------- generate normal data -------------------------
samples = 100  # 500

domains = np.zeros((samples, N, N))
for i in range(samples):
    num_circles = np.random.randint(5, 10)
    radius = np.random.uniform(0.05, 0.1)
    domains[i] = generate_circles(N, num_circles, radius)

# -------------------------------- export --------------------------------
np.save(f"../../data/graded_fibers_{N}.npy", domains)

# # ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
ax.imshow(domains[0], cmap="binary")
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.savefig(f"../../results/fibers.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


# -------------------------- generate anomalies --------------------------
# samples = 50
# num_circles = 10


# for num_squares in range(num_circles + 1):
#     domains = np.zeros((samples, N, N))
#     for i in range(samples):
#         # radius = np.random.uniform(0.02, 0.2)
#         radius = np.random.uniform(0.05, 0.1)
#         radius = 0.08
#         domains[i] = generate_squares(N, num_circles - num_squares, num_squares, radius)

# # -------------------------------- export --------------------------------
#     np.save(f"../../data/fibers_anomaly_{num_squares}_{N}.npy", domains)

# # ---------------------------- postprocessing ----------------------------
#     if num_squares == 1:
#         fig, ax = plt.subplots(figsize=(1, 1), dpi=N)
#         ax.imshow(domains[0], cmap="binary")
#         ax.axis("off")
#         fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#         plt.savefig(
#             f"../../results/fibers_anomaly.pdf", bbox_inches="tight", pad_inches=0
#         )
#         plt.show()
