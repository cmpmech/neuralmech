import numpy as np

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