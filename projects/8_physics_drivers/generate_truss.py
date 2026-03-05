import numpy as np
import matplotlib.pyplot as plt
from patterns import *

# maybe just use this?
# https://www.nature.com/articles/s41467-025-64745-9

num_nodes = 4 # at least 3
bound = 0.25 #0.25 #0.4 # for offsets (max choice should be 0.5)
# edge_prob, diag_prob = 0.8, 0.6
edge_prob, diag_prob = 1, 1
# ---------------------------- node positions ----------------------------
x = np.linspace(0, 1, num_nodes)
y = np.linspace(0, 1, num_nodes)
x, y = np.meshgrid(x, y, indexing='ij')

def fix_offsets(offsets):
    # does not prevent all overlaps
    for i in range(len(offsets) - 1):
        overlap = 1. / (num_nodes - 1) - offsets[i] + offsets[i + 1]
        if overlap < 0.1 / (num_nodes - 1):
            shift = (-overlap + 0.1 / (num_nodes - 1)) / 2.
            offsets[i] -= shift
            offsets[i + 1] += shift
    return offsets

# internal offsets
internal_offsets = np.random.uniform(-bound / (num_nodes - 1), bound / (num_nodes - 1),
                                     (2, num_nodes - 2, num_nodes - 2))
for i in range(num_nodes - 2):
    internal_offsets[0,i,:] = fix_offsets(internal_offsets[0,i,:])
    internal_offsets[1,:,i] = fix_offsets(internal_offsets[1,:,i])
x[1:-1,1:-1] += internal_offsets[0]
y[1:-1,1:-1] += internal_offsets[1]

# boundary offsets (periodic)
boundary_offsets = np.random.uniform(-bound / (num_nodes - 1), bound / (num_nodes - 1), (2, num_nodes - 2))
boundary_offsets[:,0] = np.minimum(boundary_offsets[:,0] * 2, boundary_offsets[:,0]) # boundaries have more freedom
boundary_offsets[:,-1] = np.maximum(boundary_offsets[:,-1] * 2, boundary_offsets[:,-1]) # boundaries have more freedom
boundary_offsets[0] = fix_offsets(boundary_offsets[0])
boundary_offsets[1] = fix_offsets(boundary_offsets[1])

x[1:-1,0] += boundary_offsets[0]
x[1:-1,-1] += boundary_offsets[0]
y[0,1:-1] += boundary_offsets[1]
y[-1,1:-1] += boundary_offsets[1]

x = x.flatten()
y = y.flatten()
# ----------------------------- connections ------------------------------
edges = generate_triangulated_grid(num_nodes, edge_prob, diag_prob)
# edges = generate_lattice_grid(num_nodes, edge_prob, diag_prob)
# edges = generate_warren_grid(num_nodes, edge_prob, diag_prob)

# claudes ideas
# edges = generate_radial_pattern(num_nodes)
# edges = generate_diamond_pattern(num_nodes) # useless
# edges = generate_hexagonal_inspired(num_nodes)
# edges = generate_star_pattern(num_nodes) # useless
# edges = generate_voronoi_inspired(num_nodes) # useless
# edges = generate_circular_brace(num_nodes) # useless
# edges = generate_kagome_pattern(num_nodes) # fun but useless


# probablistic removal / lowest stress removal?

# --------------------------- post-processing ----------------------------
fig, ax = plt.subplots(dpi=100)
for dx in range(3):
    for dy in range(3):
        ax.plot(np.array(x) + 1, np.array(y) + 1, 'ko', markersize=3)
        for edge in edges:
            ax.plot([x[edge[0]] + dx, x[edge[1]] + dx],
                       [y[edge[0]] + dy, y[edge[1]] + dy],
                       color='k' if dx == 1 and dy == 1 else 'gray',
                       linewidth=1 if dx == 1 and dy == 1 else 1,
                       alpha=1.0 if dx == 1 and dy == 1 else 0.4)

ax.set_aspect('equal')
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.show()

# -------------------------------- export --------------------------------
np.savez('../../data/truss.npz', x=x, y=y, edges=edges)
