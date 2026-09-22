import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.colors import to_rgba

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# physics
CONDUCTIVITY = 1.0
FLUX = 0.5  # prescribed influx on the right edge
AMPLITUDE = 50.0
SOURCE_CENTER = np.array([0.25, 0.5])
WIDTH = 0.08

# discretization
RESOLUTION = 34

# postprocessing
MESH_COLOR = to_rgba("lightgray", 0.5)
MESH_WIDTH = 1


# ---------------------------------------- helper -------------------------------------
def source(x):
    return AMPLITUDE * np.exp(-np.sum((x - SOURCE_CENTER) ** 2, axis=-1) / WIDTH**2)


def index(i, j):
    return i + RESOLUTION * j


# -------------------------------------- geometry -------------------------------------
mesh = np.load(DATA_DIR / "plate_with_hole.npz")
length = float(mesh["length"])
hole_center = mesh["hole_center"]
hole_radius = float(mesh["hole_radius"])

x = np.linspace(0, length, RESOLUTION)
y = np.linspace(0, length, RESOLUTION)
dx = x[1] - x[0]
x, y = np.meshgrid(x, y, indexing="ij")

distance = np.sum((np.stack([x, y], axis=-1) - hole_center) ** 2, axis=-1)
in_hole = distance < hole_radius**2

# -------------------------------------- assembly -------------------------------------
rows = []
cols = []
vals = []
p = np.zeros(RESOLUTION**2)
stencil = CONDUCTIVITY / dx**2

for j in range(RESOLUTION):
    for i in range(RESOLUTION):
        node = index(i, j)

        if in_hole[i, j] or i == 0 or j == 0 or j == RESOLUTION - 1:
            rows += [node]
            cols += [node]
            vals += [1.0]
        elif i == RESOLUTION - 1:
            rows += [node, node]
            cols += [node, index(i - 1, j)]
            vals += [1 / dx, -1 / dx]  # one-sided difference for grad u . n = f
            p[node] = FLUX
        else:
            rows += [node] * 5
            cols += [
                node,
                index(i + 1, j),
                index(i - 1, j),
                index(i, j + 1),
                index(i, j - 1),
            ]
            vals += [4 * stencil, -stencil, -stencil, -stencil, -stencil]
            p[node] = source(np.array([x[i, j], y[i, j]]))

A = sp.coo_matrix((vals, (rows, cols)), shape=(RESOLUTION**2,) * 2).tocsr()

print(f"{RESOLUTION} x {RESOLUTION} grid, {RESOLUTION**2} dofs")

# --------------------------------------- solve ---------------------------------------
u = spla.spsolve(A, p).reshape(RESOLUTION, RESOLUTION, order="F")
u[in_hole] = np.nan

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
cells = ax.pcolormesh(
    x,
    y,
    u,
    cmap="inferno",
    shading="nearest",
    linewidths=MESH_WIDTH,
    antialiased=True,  # pcolormesh defaults to False, which forces solid 1px lines
)
# grid lines only outside the hole: one edge color per cell, transparent where constrained
edge_colors = np.tile(MESH_COLOR, (RESOLUTION**2, 1))
edge_colors[in_hole.ravel()] = (0, 0, 0, 0)
cells.set_edgecolors(edge_colors)

# nearest shading centers a cell on each node, so the grid overhangs by dx / 2
ax.set_xlim(-dx / 2, length + dx / 2)
ax.set_ylim(-dx / 2, length + dx / 2)
ax.set_aspect("equal")
ax.axis("off")
# ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / "poisson_fdm.pdf", transparent=True)
