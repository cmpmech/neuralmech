from solvers.bouncing_balls import BouncingBalls
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

np.random.seed(0)

# ------------------------- physical parameters --------------------------
g = np.array([0, -9.81])

balls = 8

p0 = np.random.uniform(0.1, 0.9, (balls, 2))
v0 = np.random.uniform(-1, 1, (balls, 2))
R  = np.random.uniform(0.01, 0.05, balls)

bounds = [[0, 1], [0, 1]]  # x, y
e = 0.9

# ------------------------------ simulation ------------------------------
dt = 5e-3
N  = 800

solver = BouncingBalls(p0, v0, R, bounds, e=e, g=g)
p = solver.solve(dt, N)

# ---------------------------- postprocessing ----------------------------
colors = plt.cm.tab20

fig, ax = plt.subplots(dpi=150)
for j in range(0, N, 1):
    for i in range(balls):
        circle = Circle(p[j, i], R[i], fill=True, alpha=0.3, color=colors(i % colors.N))
        ax.add_patch(circle)

xmin, xmax = bounds[0]
ymin, ymax = bounds[1]
rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1)
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
rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1)
ax.add_patch(rect)
ax.set_aspect('equal')
ax.axis('off')
plt.show()

# ----------------------- animation postprocessing -----------------------
plot_every = 2
for j in range(0, N, plot_every):
    colors = plt.cm.tab20
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    for i in range(balls):
        circle = Circle(p[j, i], R[i], fill=True, alpha=0.9, color=colors(i % colors.N))
        ax.add_patch(circle)

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=4)
    ax.add_patch(rect)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.1)
    plt.savefig(f'../../results/animations/animation_frames/balls/frame_{j // plot_every}.jpg')
    plt.close()
