import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.collections import PolyCollection
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

# postprocessing
MESH_COLOR = to_rgba("lightgray", 0.5)
MESH_WIDTH = 1

TOL = 1e-6


# ---------------------------------------- helper -------------------------------------
def source(x):
    return AMPLITUDE * np.exp(-np.sum((x - SOURCE_CENTER) ** 2, axis=-1) / WIDTH**2)


# ------------------------------------- load mesh -------------------------------------
mesh = np.load(DATA_DIR / "plate_with_hole.npz")
coords = mesh["coords"]
quads = mesh["quads"]
length = float(mesh["length"])

# ----------------------------------- preprocessing -----------------------------------
centroids = coords[quads].mean(axis=1)

x, y = coords[quads][:, :, 0], coords[quads][:, :, 1]
shoelace = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
areas = 0.5 * np.abs(shoelace.sum(axis=1))

faces = {}
for cell, quad in enumerate(quads):
    for i in range(4):
        faces.setdefault(tuple(sorted((quad[i], quad[(i + 1) % 4]))), []).append(cell)

on_right = coords[:, 0] > length - TOL

# -------------------------------------- assembly -------------------------------------
rows = []
cols = []
vals = []
f = areas * source(centroids)

for (a, b), cells in faces.items():
    face_length = np.linalg.norm(coords[b] - coords[a])

    if len(cells) == 2:
        i, j = cells
        dist = np.linalg.norm(centroids[i] - centroids[j])
        coeff = CONDUCTIVITY * face_length / dist
        rows += [i, i, j, j]
        cols += [i, j, j, i]
        vals += [coeff, -coeff, coeff, -coeff]
    elif on_right[a] and on_right[b]:
        f[cells[0]] += CONDUCTIVITY * FLUX * face_length
    else:
        i = cells[0]
        dist = np.linalg.norm(centroids[i] - 0.5 * (coords[a] + coords[b]))
        rows += [i]
        cols += [i]
        vals += [CONDUCTIVITY * face_length / dist]  # dirichlet value is zero

K = sp.coo_matrix((vals, (rows, cols)), shape=(len(quads), len(quads))).tocsr()

print(f"{len(quads)} cells, {len(faces)} faces, {len(quads)} dofs")

# --------------------------------------- solve ---------------------------------------
u = spla.spsolve(K, f)

# ----------------------------------- postprocessing ----------------------------------
patches = PolyCollection(
    coords[quads], cmap="inferno", edgecolors=MESH_COLOR, linewidths=MESH_WIDTH
)
patches.set_array(u)

fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
ax.add_collection(patches)

ax.set_xlim(0, length)
ax.set_ylim(0, length)
ax.set_aspect("equal")
ax.axis("off")
# ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / "poisson_fvm.pdf", transparent=True)
