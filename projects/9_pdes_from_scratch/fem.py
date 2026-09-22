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

# quadrature
GAUSS_POINTS = np.array([-1.0, 1.0]) / np.sqrt(3)
GAUSS_WEIGHTS = np.array([1.0, 1.0])

# postprocessing
MESH_COLOR = to_rgba("lightgray", 0.5)
MESH_WIDTH = 1

TOL = 1e-6


# ---------------------------------------- helper -------------------------------------
def source(x):
    return AMPLITUDE * np.exp(-np.sum((x - SOURCE_CENTER) ** 2, axis=-1) / WIDTH**2)


def shape_functions(xi, eta):
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    N = 0.25 * (1 + corners[:, 0] * xi) * (1 + corners[:, 1] * eta)
    dN_dxi = 0.25 * np.stack(
        [
            corners[:, 0] * (1 + corners[:, 1] * eta),
            corners[:, 1] * (1 + corners[:, 0] * xi),
        ]
    )
    return N, dN_dxi


# ------------------------------------- load mesh -------------------------------------
mesh = np.load(DATA_DIR / "plate_with_hole.npz")
coords = mesh["coords"]
quads = mesh["quads"]
length = float(mesh["length"])

# --------------------------------- boundary conditions -------------------------------
edges = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)])
_, ids, counts = np.unique(
    np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True
)
boundary_edges = edges[ids[counts == 1]]

on_boundary = np.zeros(len(coords), dtype=bool)
on_boundary[boundary_edges] = True
on_left = coords[:, 0] < TOL
on_right = coords[:, 0] > length - TOL
on_bottom = coords[:, 1] < TOL
on_top = coords[:, 1] > length - TOL
on_hole = on_boundary & ~(on_left | on_right | on_bottom | on_top)

fixed = on_left | on_bottom | on_top | on_hole
free = ~fixed
neumann_edges = boundary_edges[on_right[boundary_edges].all(axis=1)]

# -------------------------------------- assembly -------------------------------------
rows = []
cols = []
vals = []
f = np.zeros(len(coords))

for quad in quads:
    xe = coords[quad]
    Ke = np.zeros((4, 4))
    fe = np.zeros(4)
    for xi, wxi in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        for eta, weta in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            N, dN_dxi = shape_functions(xi, eta)
            jac = dN_dxi @ xe  # jac[b, a] = dx_a / dxi_b
            dN_dx = np.linalg.solve(jac, dN_dxi)
            weight = np.linalg.det(jac) * wxi * weta

            Ke += CONDUCTIVITY * dN_dx.T @ dN_dx * weight
            fe += N * source(N @ xe) * weight

    rows.append(np.repeat(quad, 4))
    cols.append(np.tile(quad, 4))
    vals.append(Ke.ravel())
    f[quad] += fe

for edge in neumann_edges:
    xe = coords[edge]
    detj = 0.5 * np.linalg.norm(xe[1] - xe[0])
    for xi, w in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
        f[edge] += N * CONDUCTIVITY * FLUX * detj * w

K = sp.coo_matrix(
    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
    shape=(len(coords), len(coords)),
).tocsr()

print(f"{len(quads)} elements, {len(coords)} nodes, {free.sum()} dofs")

# --------------------------------------- solve ---------------------------------------
u = np.zeros(len(coords))
u[free] = spla.spsolve(K[free][:, free], f[free])

# ----------------------------------- postprocessing ----------------------------------
triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
ax.tripcolor(
    coords[:, 0], coords[:, 1], triangles, u, shading="gouraud", cmap="inferno"
)
ax.add_collection(
    PolyCollection(
        coords[quads], facecolors="none", edgecolors=MESH_COLOR, linewidths=MESH_WIDTH
    )
)

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
    plt.savefig(RGB_PDF_DIR / "poisson_fem.pdf", transparent=True)
