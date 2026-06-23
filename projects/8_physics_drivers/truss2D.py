from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from solvers.truss import global_stiffness_matrix

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# geometry
NUM_BAYS = 9  # bottom-chord panels (>= 3)
SPAN = 1.0
RISE = 0.3  # crown height of the parabolic top chord

# physics
EA = 1.0
LOAD = 0.01  # downward point load per interior bottom node

# postprocessing
SCALING = 0.5  # deformation magnification


# ---------------------------------------- helper -------------------------------------
def generate_arch_truss(num_bays, span, rise):
    dx = span / num_bays
    bottom = [(i * dx, 0.0) for i in range(num_bays + 1)]
    top = []
    for i in range(1, num_bays):
        xi = i * dx
        top.append((xi, rise * (1.0 - ((xi - span / 2.0) / (span / 2.0)) ** 2)))
    coords = np.array(bottom + top)

    def top_id(i):
        return num_bays + i  # top node sitting above bottom node i

    edges = []
    for i in range(num_bays):  # bottom chord
        edges.append([i, i + 1])
    for i in range(1, num_bays - 1):  # top chord
        edges.append([top_id(i), top_id(i + 1)])
    for i in range(1, num_bays):  # verticals
        edges.append([i, top_id(i)])
    for i in range(1, num_bays - 1):  # interior cross bracing
        edges.append([i, top_id(i + 1)])
        edges.append([i + 1, top_id(i)])
    edges.append([0, top_id(1)])  # end diagonals
    edges.append([num_bays, top_id(num_bays - 1)])

    pinned = 0  # left support, fixed in x and y
    roller = num_bays  # right support, fixed in y
    loaded = list(range(1, num_bays))  # interior bottom nodes
    return coords, np.array(edges), pinned, roller, loaded


# ----------------------------------- preprocessing -----------------------------------
coords, edges, pinned, roller, loaded = generate_arch_truss(NUM_BAYS, SPAN, RISE)

lengths = np.zeros(len(edges))
rotations = np.zeros(len(edges))
for i, edge in enumerate(edges):
    dist = coords[edge[1]] - coords[edge[0]]
    lengths[i] = np.linalg.norm(dist)
    rotations[i] = np.arctan2(dist[1], dist[0])

# ----------------------------------- setup solver ------------------------------------
K = global_stiffness_matrix(EA, edges, lengths, rotations, 2 * len(coords))
F = np.zeros(2 * len(coords))
for node in loaded:
    F[2 * node + 1] = -LOAD

# ------------------------------- boundary conditions ---------------------------------
constrained = [2 * pinned, 2 * pinned + 1, 2 * roller + 1]
for dof in constrained:
    K[dof, :] = 0.0
    K[:, dof] = 0.0
    K[dof, dof] = 1.0
    F[dof] = 0.0

# --------------------------------------- solve ---------------------------------------
U = np.linalg.solve(K, F).reshape(-1, 2)

# ----------------------------------- postprocessing ----------------------------------
deformedcoords = coords + U * SCALING
disp_y = U[:, 1]

fig, ax = plt.subplots(dpi=100)
cmap = cm.turbo
norm = colors.Normalize(vmin=disp_y.min(), vmax=disp_y.max())
for edge in edges:
    ax.plot(
        [coords[edge[0], 0], coords[edge[1], 0]],
        [coords[edge[0], 1], coords[edge[1], 1]],
        color="k",
        linewidth=1,
        alpha=0.1,
    )
    ax.plot(
        [deformedcoords[edge[0], 0], deformedcoords[edge[1], 0]],
        [deformedcoords[edge[0], 1], deformedcoords[edge[1], 1]],
        color=cmap(norm(disp_y[edge].mean())),
        linewidth=2,
    )
ax.plot(deformedcoords[:, 0], deformedcoords[:, 1], "ko", markersize=4)

ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
