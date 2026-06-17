from solvers.inverted_pendulum import CartPole
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

# -------------------------- problem definition --------------------------
# cart-pole parameters
M = 1.0   # cart mass
m = 0.1   # pole mass
l = 1.0   # pole length
g = 9.81
d_cart, d_pole = 0.0, 0.0

# control input u = f(t, Y); open-loop for now (no actuation)
f = lambda t, Y : 0.0
# later (balance upright): f = lambda t, Y : -K @ (np.array(Y) - Y_target)

# initial condition: small tilt from upright (theta measured from vertical)
Y0 = [0.0, 0.1, 0.0, 0.0]  # [x, theta, dx, dtheta]
T = 5.0
dt = 0.01

# -------------------------------- solve ---------------------------------
solver = CartPole(M, m, l, g, f, d_cart=d_cart, d_pole=d_pole)
t, Y = solver.solve(Y0, T, dt)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(t, Y[:, 0], 'k')  # cart position
ax.plot(t, Y[:, 1], 'r')  # pole angle
plt.show()

# ----------------------- animation postprocessing -----------------------
cart_w, cart_h = 0.3, 0.2
xpad = l + cart_w

plot_every = 2
for j in range(0, len(t), plot_every):
    x, th = Y[j, 0], Y[j, 1]
    pivot = np.array([x, cart_h])
    bob = pivot + l * np.array([np.sin(th), np.cos(th)])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    cart = Rectangle((x - cart_w / 2, 0), cart_w, cart_h, fill=True, color='k')
    ax.add_patch(cart)
    ax.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color='r', linewidth=2)
    ax.add_patch(Circle(bob, 0.05, fill=True, color='r'))
    ax.plot([Y[:, 0].min() - xpad, Y[:, 0].max() + xpad], [0, 0], 'k', linewidth=1)

    ax.set_xlim(Y[:, 0].min() - xpad, Y[:, 0].max() + xpad)
    ax.set_ylim(-0.3, l + cart_h + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.1)
    plt.savefig(f'../../results/animations/animation_frames/pendulum/frame_{j // plot_every}.jpg')
    plt.close()
