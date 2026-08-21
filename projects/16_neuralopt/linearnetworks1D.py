"""Loss landscape of a two-layer linear network under a nonconvex scalar cost.

Network:  s = w2 . w1        (w1, w2 in R^WIDTH, scalar input x = 1)
Loss:     L = COST(s)

The loss depends on the 2*WIDTH parameters only through the product s, so the
landscape is the pullback of COST along (w1, w2) -> w2 . w1.  In the aligned
coordinates

    p = (w1 + w2)/2,   m = (w1 - w2)/2,   a = |p| + |m|,   b = |p| - |m|

one has a*b = s exactly and a^2 - b^2 conserved under gradient flow, for any
WIDTH and any COST.  The (a, b) plane below is therefore not a slice: it is the
full system.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

# ------------------------------------------------------------------ knobs
COST = lambda y: np.cos(y) * y**2
DCOST = lambda y: -np.sin(y) * y**2 + 2.0 * y * np.cos(y)

WIDTH = 8  # hidden width r, so 2*r parameters
RANGE = 4.0  # half-range of the (a, b) plot
GRID = 601  # grid resolution
NTRAJ = 4  # gradient-flow trajectories
LR = 2e-3  # step size
STEPS = 6000
INIT = 0.9  # init scale of w1, w2
SEED = 0

# ------------------------------------------------------------- landscape
ax = np.linspace(-RANGE, RANGE, GRID)
A, B = np.meshgrid(ax, ax)
L = COST(A * B)

# ------------------------------------------- stationary branches a*b = s*
s = np.linspace(-(RANGE**2), RANGE**2, 200001)
d = DCOST(s)
roots = s[:-1][np.sign(d[:-1]) != np.sign(d[1:])]
roots = roots[np.abs(roots) > 1e-9]

# ------------------------------------------ gradient flow in 2*WIDTH dims
rng = np.random.default_rng(SEED)
w1 = rng.normal(scale=INIT, size=(NTRAJ, WIDTH))
w2 = rng.normal(scale=INIT, size=(NTRAJ, WIDTH))

traj = np.empty((STEPS + 1, NTRAJ, 2))


def project(w1, w2):
    """Map the 2*WIDTH parameters onto the exact (a, b) plane."""
    P = np.linalg.norm(0.5 * (w1 + w2), axis=1)
    M = np.linalg.norm(0.5 * (w1 - w2), axis=1)
    return np.stack([P + M, P - M], axis=1)


traj[0] = project(w1, w2)
for k in range(STEPS):
    g = DCOST(np.sum(w1 * w2, axis=1))[:, None]
    w1, w2 = w1 - LR * g * w2, w2 - LR * g * w1
    traj[k + 1] = project(w1, w2)

# ------------------------------------------------------------- self-check
a, b = traj[-1, :, 0], traj[-1, :, 1]
err_prod = np.max(np.abs(a * b - np.sum(w1 * w2, axis=1)))
err_cons = np.max(
    np.abs(
        (traj[:, :, 0] ** 2 - traj[:, :, 1] ** 2)
        - (traj[0, :, 0] ** 2 - traj[0, :, 1] ** 2)
    )
)
print(f"max |a*b - s|           = {err_prod:.2e}")
print(f"max drift of a^2 - b^2  = {err_cons:.2e}")

# ------------------------------------------------------------------ plot
fig, (axL, axC) = plt.subplots(1, 2, figsize=(12, 5.4), width_ratios=[1.35, 1])

norm = SymLogNorm(linthresh=1.0, vmin=L.min(), vmax=L.max())
axL.pcolormesh(A, B, L, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
axL.contour(A, B, L, levels=30, colors="k", linewidths=0.3, alpha=0.35)

t = np.linspace(-RANGE, RANGE, 2000)
t = t[np.abs(t) > 1e-3]
for r in roots:
    for sg in (1, -1):
        y = r / (sg * t)
        keep = np.abs(y) <= RANGE
        axL.plot(sg * t[keep], y[keep], color="darkorange", lw=1.4, zorder=3)

for delta in (0.25, 1.0, 2.25, 4.0, 6.25):
    u = np.linspace(-2.4, 2.4, 400)
    c, sh = np.sqrt(delta) * np.cosh(u), np.sqrt(delta) * np.sinh(u)
    for sx, sy in ((1, 1), (-1, 1)):
        axL.plot(sx * c, sy * sh, color="0.35", lw=0.7, ls="--", alpha=0.6)
        axL.plot(sy * sh, sx * c, color="0.35", lw=0.7, ls="--", alpha=0.6)

for i in range(NTRAJ):
    axL.plot(traj[:, i, 0], traj[:, i, 1], color="k", lw=1.6, zorder=4)
    axL.plot(*traj[0, i], "o", ms=4, mfc="w", mec="k", zorder=5)
    axL.plot(*traj[-1, i], "*", ms=11, color="k", zorder=5)

axL.set(
    xlim=(-RANGE, RANGE),
    ylim=(-RANGE, RANGE),
    xlabel="a",
    ylabel="b",
    title=f"width r = {WIDTH}  ({2 * WIDTH} parameters)",
)
axL.set_aspect("equal")

sp = np.linspace(-(RANGE**2), RANGE**2, 4000)
axC.plot(sp, COST(sp), color="C0", lw=1.5)
axC.plot(roots, COST(roots), "o", ms=4, color="darkorange")
axC.axhline(0, color="0.6", lw=0.5)
axC.set(xlabel="s = a*b", ylabel="C(s)", title="cost on the effective parameter")

fig.tight_layout()
plt.show()
