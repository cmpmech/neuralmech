import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

np.random.seed(0)

# ------------------------- physical parameters --------------------------
g = np.array([0, -9.81])

balls = 8

# initial conditions
p0 = np.random.uniform(0.1, 0.9, (balls, 2))
v0 = np.random.uniform(-1, 1, (balls, 2))
R = np.random.uniform(0.01,0.05, balls)


# debugging
# g = np.array([0, 0])
# p0 = np.array([[0.1, 0.5], [0.9, 0.45]])
# v0 = np.array([[1, 0], [-1, 0]])
# R = np.array([0.05, 0.01])

m = R**2

bounds = [[0, 1], [0, 1]] # x, y

e = 0.9 #0.9
# ------------------------------ simulation ------------------------------
dt = 5e-3 #1e-2
N = 800 # 200
# N = 200 #80 #200

p = np.zeros((N + 1, balls, 2))

v = v0
p[0] = p0
for n in range(N):
    # symplectic Euler
    v = v + g * dt
    p[n + 1] = p[n] + v * dt

    # boundary contact
    left = (p[n + 1, :, 0] < bounds[0][0] + R) #& (v[:, 0] < 0)
    right = (p[n + 1, :, 0] > bounds[0][1] - R) #& (v[:, 0] > 0)
    p[n + 1, left, 0] = bounds[0][0] + R[left]
    p[n + 1, right, 0] = bounds[0][1] - R[right]
    v[left, 0] *= -e
    v[right, 0] *= -e

    bot = (p[n + 1, :, 1] < bounds[1][0] + R) #& (v[:, 1] < 0)
    top = (p[n + 1, :, 1] > bounds[1][1] - R) #& (v[:, 1] > 0)
    p[n + 1, bot, 1] = bounds[1][0] + R[bot]
    p[n + 1, top, 1] = bounds[1][1] - R[top]
    v[bot, 1] *= -e
    v[top, 1] *= -e

    # ball-ball contact
    for i in range(balls):
        for j in range(i + 1, balls):
            dp = p[n + 1, i] - p[n + 1, j]
            dist = np.linalg.norm(dp)
            min_dist = R[i] + R[j]

            if dist < min_dist:
                n_ij = dp / (dist + 1e-12)
                dv = v[i] - v[j]
                vn = np.dot(dv, n_ij)

                if vn < 0:
                    mi, mj = m[i], m[j]

                    J = -(1 + e) * vn / (1 / mi + 1 / mj)

                    v[i] += (J / mi) * n_ij
                    v[j] -= (J / mj) * n_ij

                    # positional correction (mass-weighted)
                    overlap = min_dist - dist
                    p[n + 1, i] += overlap * (mj / (mi + mj)) * n_ij
                    p[n + 1, j] -= overlap * (mi / (mi + mj)) * n_ij
# --------------------------- post-processing ----------------------------
colors = plt.cm.tab20
fig, ax = plt.subplots(dpi=150)
for j in range(0, N, 1):
    for i in range(balls):
        circle = Circle(p[j, i], R[i], fill=True, alpha=0.3, color=colors(i % colors.N))
        ax.add_patch(circle)

xmin, xmax = bounds[0]
ymin, ymax = bounds[1]
rect = Rectangle((xmin, ymin), xmax - xmin,
                 ymax - ymin, fill=False, linewidth=1)

ax.add_patch(rect)
ax.set_aspect('equal')
ax.axis('off')
plt.show()

fig, ax = plt.subplots(dpi=150)
for j in range(0, N, 10):
    for i in range(balls):
        circle = Circle(p[j, i], R[i], fill=True, alpha=0.1, color=colors(i % colors.N))
        ax.add_patch(circle)

for i in range(balls):
    ax.plot(p[:, i, 0], p[:, i, 1], color=colors(i % colors.N))


xmin, xmax = bounds[0]
ymin, ymax = bounds[1]
rect = Rectangle((xmin, ymin), xmax - xmin,
                 ymax - ymin, fill=False, linewidth=1)

ax.add_patch(rect)
ax.set_aspect('equal')
ax.axis('off')
plt.show()


# ------------------------------- animate --------------------------------
plot_every = 2
for j in range(0, N, plot_every):
    colors = plt.cm.tab20
    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
    for i in range(balls):
        circle = Circle(p[j, i], R[i], fill=True, alpha=0.9, color=colors(i % colors.N))
        ax.add_patch(circle)

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    rect = Rectangle((xmin, ymin), xmax - xmin,
                     ymax - ymin, fill=False, linewidth=4)
    ax.add_patch(rect)
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_rasterized(True)
    fig.tight_layout(pad=0.1)
    plt.savefig(f'animation_frames/balls_{j // plot_every}.png')
    plt.close()



# import numpy as np
# import matplotlib.pyplot as plt
#
# # ------------------------- physical parameters --------------------------
# g = np.array([0, -9.81])
#
# # initial conditions
# p0 = np.random.uniform(0.1, 0.9, 2)
# v0 = np.random.uniform(0.1, 0.9, 2)
# R = np.random.uniform(0.01, 0.1)
#
# bounds = [[0, 1], [0, 1]]  # x, y
#
# e = 0.8
# # ------------------------------ simulation ------------------------------
# dt = 1e-2
# N = 200
#
# p = np.zeros((N + 1, 2))
#
# v = v0
# p[0] = p0
# for n in range(N):
#     # symplectic Euler
#     v = v + g * dt
#     p[n + 1] = p[n] + v * dt
#
#     # boundary check
#     if p[n + 1, 0] < bounds[0][0] + R and v[0] < 0:
#         p[n + 1, 0] = bounds[0][0] + R
#         v[0] = -e * v[0]
#     elif p[n + 1, 0] > bounds[0][1] - R and v[0] > 0:
#         p[n + 1, 0] = bounds[0][1] - R
#         v[0] = -e * v[0]
#     if p[n + 1, 1] < bounds[1][0] + R and v[1] < 0:
#         p[n + 1, 1] = bounds[1][0] + R
#         v[1] = -e * v[1]
#     elif p[n + 1, 1] > bounds[1][1] - R and v[1] > 0:
#         p[n + 1, 1] = bounds[1][1] - R
#         v[1] = -e * v[1]
#
# # --------------------------- post-processing ----------------------------
# fig, ax = plt.subplots()
# ax.scatter(p[:, 0], p[:, 1], c='r', s=1)
# plt.show()






