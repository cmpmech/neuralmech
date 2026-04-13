import cmasher as cmr
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["axes.linewidth"] = 2.5
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], "size": 22})
rc("text", usetex=True)

# --------------------------- softmax sampling ---------------------------
x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)
x, y = np.meshgrid(x, y, indexing="ij")

z = np.exp(x) / (np.exp(x) + np.exp(y))
print(np.max(z), np.min(z))

# ----------------------- book postpostprocessing ------------------------
fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
cb = ax.pcolormesh(x, y, z, cmap="cividis")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f"../../results/softmax.png", bbox_inches="tight", pad_inches=0)
plt.show()

fig, ax = plt.subplots(figsize=(0.7, 3), dpi=100)  # Adjust height for colorbar
ax.set_visible(False)
cbar = fig.colorbar(
    cb, ax=ax, fraction=1.0, pad=0.04, format="%.1f", location="right", aspect=15
)
cbar.ax.tick_params(width=1)
cbar.outline.set_visible(False)
fig.tight_layout(pad=0)
ax.set_rasterized(True)
plt.savefig(
    "../../results/softmax_colorbar.pdf",
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()
