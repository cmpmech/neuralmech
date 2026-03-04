import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
import matplotlib.colors as colors

# -------------------------------- helper --------------------------------
def local_stiffness_matrix(EA, length, rotation):
    c = np.cos(rotation)
    s = np.sin(rotation)
    Ke = np.array([[c**2, s*c, -c**2, -s*c],
                   [s*c, s**2, -s*c, -s**2],
                   [-c**2, -s*c, c**2, s*c],
                   [-s*c, -s**2, s*c, s**2]])
    return EA / length * Ke

def edge_to_eft(edge):
    eft = np.zeros(2 * len(edge), dtype=np.int32)
    eft[::2] = edge * 2
    eft[1::2] = edge * 2 + 1
    return eft

def global_stiffness_matrix(EA, edges, lengths, rotations, dofs):
    K = np.zeros((dofs, dofs))
    for i in range(len(edges)):
        Ke = local_stiffness_matrix(EA, lengths[i], rotations[i])
        eft = edge_to_eft(edges[i])
        K[np.ix_(eft, eft)] += Ke
    return K

def multi_freedom_constraint(K, F, i, j, c, w=1e12):
    K[i, i] += w
    K[j, j] += w
    K[i, j] -= w
    K[j, i] -= w
    F[i] += w * c
    F[j] -= w * c
    return K, F

def get_strains(edges, U, coords, lengths):
    strains = np.zeros(len(edges))
    for i in range(len(edges)):
        u1 = U[edges[i][0]]
        u2 = U[edges[i][1]]
        x1 = coords[edges[i][0]]
        x2 = coords[edges[i][1]]

        du = u2 - u1
        t = (x2 - x1) / lengths[i]  # unit length
        strains[i] = (du @ t) / lengths[i]
    return strains

# ---------------------------- pre-processing ----------------------------
structure = np.load('truss.npz')
coords = np.vstack([structure['x'], structure['y']]).T
edges = structure['edges']
num_nodes = int(np.sqrt(len(coords)))

lengths = np.zeros(len(edges))
rotations = np.zeros(len(edges))
for i, edge in enumerate(edges): # SHOULD BE DIRECTED (NOT DOUBLED)
    dist = coords[edge[1]] - coords[edge[0]]
    lengths[i] = np.linalg.norm(dist)
    if dist[0] != 0:
        rotations[i] = np.atan(dist[1] / dist[0])
    else:
        rotations[i] = np.asin(np.sign(dist[1]))

# ------------------------------- physics --------------------------------
EA = 1.

# ----------------------------- setup solver -----------------------------
tic = time.time()
K = global_stiffness_matrix(EA, edges, lengths, rotations, 2*len(coords))
F = np.zeros(2*len(coords))
print(f'dofs: {2*len(coords):d}')
# ------------------------- boundary conditions --------------------------
left_edge = edge_to_eft(np.array([i for i in range(num_nodes)]))
right_edge = edge_to_eft(np.array([(num_nodes - 1) * num_nodes + i for i in range(num_nodes)]))
bot_edge = edge_to_eft(np.array([i * num_nodes for i in range(num_nodes)]))
top_edge = edge_to_eft(np.array([i * num_nodes + (num_nodes - 1) for i in range(num_nodes)]))

left_top_corner = left_edge[-2:]
left_bot_corner = left_edge[:2]
right_top_corner = right_edge[-2:]
right_bot_corner = right_edge[:2]

K_full = K.copy() # for reaction force
F_full = F.copy() # for reaction force

w = 1e12
# left right x
strainxx = 1.
K, F = multi_freedom_constraint(K, F, right_edge[0::2],
                                left_edge[0::2], strainxx * 1., w)

# left right y
K, F = multi_freedom_constraint(K, F, left_edge[1::2],
                                right_edge[1::2], 0., w)
# bot top x
K, F = multi_freedom_constraint(K, F, bot_edge[0::2],
                                top_edge[0::2], 0., w)
# bot top y
K, F = multi_freedom_constraint(K, F, bot_edge[1::2],
                                top_edge[1::2], 0., w)
# constrain left bot corner
K[left_bot_corner, :] = 0.
K[:, left_bot_corner] = 0.
K[left_bot_corner, left_bot_corner] = 1.
F[left_bot_corner] = 0.

toc = time.time()
print(f'assembly time: {(toc - tic)*1e3:.2f}ms')

# -------------------------------- solve ---------------------------------
tic = time.time()
U = np.linalg.solve(K, F)
toc = time.time()
print(f'solve time: {(toc - tic)*1e3:.2f}ms')
R = K_full@U - F_full

F_eff = np.sum(R[right_edge[0::2]])
E_eff = F_eff / strainxx # assuming A = 1.
print(f'effective stiffness E_x: {E_eff:.2f}')

U = U.reshape(-1, 2)
strains = get_strains(edges, U, coords, lengths)

# --------------------------- post-processing ----------------------------
scaling = 0.1
deformedcoords = coords + U * scaling

fig, ax = plt.subplots(dpi=100)
ax.plot(coords[:,0], coords[:,1], 'o', color='gray', markersize=4)
ax.plot(deformedcoords[:,0], deformedcoords[:,1], 'ko', markersize=4)

cmap = matplotlib.colormaps['coolwarm']
norm = colors.Normalize(vmin=-max(abs(strains)), vmax=max(abs(strains)))
for strain, edge in zip(strains, edges):
    # undeformed
    ax.plot([coords[:,0][edge[0]], coords[:,0][edge[1]]],
               [coords[:,1][edge[0]], coords[:,1][edge[1]]],
               color='k', linewidth=1, alpha=0.1)
    # deformed
    ax.plot([deformedcoords[:,0][edge[0]], deformedcoords[:,0][edge[1]]],
               [deformedcoords[:,1][edge[0]], deformedcoords[:,1][edge[1]]],
               color=cmap(norm(strain)), linewidth=2)

ax.set_aspect('equal')
plt.show()