import numpy as np

def get_node_index(i, j, num_nodes):
    # convert 2D grid indices to flat array index
    return i * num_nodes + j

def generate_triangulated_grid(num_nodes, edge_prob=0.8, diag_prob=0.6):
    # adjacency is always the same
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # horizontal and vertical
            if j < num_nodes - 1 and np.random.rand() < edge_prob:
                edges.append([idx, get_node_index(i, j + 1, num_nodes)])
            if i < num_nodes - 1 and np.random.rand() < edge_prob:
                edges.append([idx, get_node_index(i + 1, j, num_nodes)])

            # one diagonal per cell
            if i < num_nodes - 1 and j < num_nodes - 1 and np.random.rand() < diag_prob:
                edges.append([idx, get_node_index(i + 1, j + 1, num_nodes)])
    return edges

def generate_lattice_grid(num_nodes, edge_prob=0.8, diag_prob=0.6):
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # all possible connections to neighbors
            if j < num_nodes - 1 and np.random.rand() < edge_prob:
                edges.append([idx, get_node_index(i, j + 1, num_nodes)])
            if i < num_nodes - 1 and np.random.rand() < edge_prob:
                edges.append([idx, get_node_index(i + 1, j, num_nodes)])
            if i < num_nodes - 1 and j < num_nodes - 1 and np.random.rand() < diag_prob:
                edges.append([idx, get_node_index(i + 1, j + 1, num_nodes)])
            if i < num_nodes - 1 and j > 0 and np.random.rand() < diag_prob:
                edges.append([get_node_index(i, j, num_nodes),
                              get_node_index(i + 1, j - 1,num_nodes)])
    return edges

def generate_warren_grid(num_nodes, edge_prob=0.8, diag_prob=0.6):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)
            # horizontal
            if j < num_nodes - 1 and np.random.rand() < edge_prob:
                edges.append([idx, get_node_index(i, j + 1, num_nodes)])
            # vertical
            if i < num_nodes - 1 and np.random.rand() < edge_prob:
                edges.append([idx, get_node_index(i + 1, j, num_nodes)])

            # diagonal (alternating)
            if i < num_nodes - 1 and j < num_nodes - 1 and np.random.rand() < diag_prob:
                if (i + j) % 2 == 0:
                    edges.append([idx, get_node_index(i + 1, j + 1, num_nodes)])
                else:
                    edges.append([get_node_index(i, j + 1, num_nodes),
                                  get_node_index(i + 1, j, num_nodes)])
    return edges

# claudes ideas
def generate_radial_pattern(num_nodes):
    """Radial pattern emanating from center"""
    edges = []
    center = (num_nodes - 1) / 2.0

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Basic grid
            if j < num_nodes - 1:
                edges.append([idx, get_node_index(i, j + 1, num_nodes)])
            if i < num_nodes - 1:
                edges.append([idx, get_node_index(i + 1, j, num_nodes)])

            # Radial diagonals based on position relative to center
            if i < num_nodes - 1 and j < num_nodes - 1:
                # Distance from center
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)

                # Connect outward from center
                if i <= center and j <= center:  # Top-left quadrant
                    edges.append(
                        [get_node_index(i + 1, j + 1, num_nodes), idx])
                elif i <= center and j > center:  # Top-right quadrant
                    edges.append([get_node_index(i + 1, j, num_nodes),
                                  get_node_index(i, j + 1, num_nodes)])
                elif i > center and j <= center:  # Bottom-left quadrant
                    edges.append([get_node_index(i, j + 1, num_nodes),
                                  get_node_index(i + 1, j, num_nodes)])
                else:  # Bottom-right quadrant
                    edges.append(
                        [idx, get_node_index(i + 1, j + 1, num_nodes)])

    return edges


def generate_diamond_pattern(num_nodes):
    """Diamond/rhombus pattern with 45-degree rotation symmetry"""
    edges = []
    center = (num_nodes - 1) / 2.0

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Create diamond shapes
            manhattan_dist = abs(i - center) + abs(j - center)

            # Horizontal connections (selective based on diamond layers)
            if j < num_nodes - 1:
                if manhattan_dist % 2 == 0 or abs(i - center) < 1:
                    edges.append(
                        [idx, get_node_index(i, j + 1, num_nodes)])

            # Vertical connections
            if i < num_nodes - 1:
                if manhattan_dist % 2 == 0 or abs(j - center) < 1:
                    edges.append(
                        [idx, get_node_index(i + 1, j, num_nodes)])

            # Diagonal connections forming diamonds
            if i < num_nodes - 1 and j < num_nodes - 1:
                if manhattan_dist % 2 == 0:
                    edges.append(
                        [idx, get_node_index(i + 1, j + 1, num_nodes)])
                    edges.append([get_node_index(i, j + 1, num_nodes),
                                  get_node_index(i + 1, j, num_nodes)])

    return edges


def generate_hexagonal_inspired(num_nodes):
    """Hexagonal-inspired pattern adapted to square grid"""
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Create hexagon-like connectivity
            # Every other row is offset
            if i % 2 == 0:
                # Standard connections
                if j < num_nodes - 1:
                    edges.append(
                        [idx, get_node_index(i, j + 1, num_nodes)])
                if i < num_nodes - 1:
                    edges.append(
                        [idx, get_node_index(i + 1, j, num_nodes)])
                    if j > 0:
                        edges.append([idx, get_node_index(i + 1, j - 1,
                                                          num_nodes)])
            else:
                # Offset connections
                if j < num_nodes - 1:
                    edges.append(
                        [idx, get_node_index(i, j + 1, num_nodes)])
                if i < num_nodes - 1:
                    edges.append(
                        [idx, get_node_index(i + 1, j, num_nodes)])
                    if j < num_nodes - 1:
                        edges.append([idx, get_node_index(i + 1, j + 1,
                                                          num_nodes)])

    return edges


def generate_star_pattern(num_nodes):
    """Star/asterisk pattern with 8-fold symmetry"""
    edges = []
    center = (num_nodes - 1) / 2.0

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Radial and circumferential connections
            di = i - center
            dj = j - center
            angle = np.arctan2(di, dj)

            # Main grid
            if j < num_nodes - 1:
                edges.append([idx, get_node_index(i, j + 1, num_nodes)])
            if i < num_nodes - 1:
                edges.append([idx, get_node_index(i + 1, j, num_nodes)])

            # Star rays (8 directions)
            if i < num_nodes - 1 and j < num_nodes - 1:
                # Determine which ray sector this cell belongs to
                sector = int((angle + np.pi) / (np.pi / 4)) % 8

                if sector in [0, 4]:  # Vertical rays
                    if abs(dj) < 0.5:
                        edges.append([idx, get_node_index(i + 1, j + 1,
                                                          num_nodes)])
                elif sector in [2, 6]:  # Horizontal rays
                    if abs(di) < 0.5:
                        edges.append([idx, get_node_index(i + 1, j + 1,
                                                          num_nodes)])
                elif sector in [1, 3, 5, 7]:  # Diagonal rays
                    edges.append(
                        [idx, get_node_index(i + 1, j + 1, num_nodes)])

    return edges


def generate_voronoi_inspired(num_nodes, seed=42):
    """Organic Voronoi-like pattern"""
    np.random.seed(seed)
    edges = []

    # Create "seed points" for Voronoi regions
    n_seeds = 5
    seeds = np.random.rand(n_seeds, 2) * (num_nodes - 1)

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Find closest seed
            distances = [np.sqrt((i - s[0]) ** 2 + (j - s[1]) ** 2) for s
                         in seeds]
            closest_seed = np.argmin(distances)

            # Connect to neighbors in same or adjacent Voronoi cell
            if j < num_nodes - 1:
                j_distances = [
                    np.sqrt((i - s[0]) ** 2 + (j + 1 - s[1]) ** 2) for s
                    in seeds]
                if np.argmin(j_distances) in [closest_seed, (
                                                                    closest_seed + 1) % n_seeds]:
                    edges.append(
                        [idx, get_node_index(i, j + 1, num_nodes)])

            if i < num_nodes - 1:
                i_distances = [
                    np.sqrt((i + 1 - s[0]) ** 2 + (j - s[1]) ** 2) for s
                    in seeds]
                if np.argmin(i_distances) in [closest_seed, (
                                                                    closest_seed + 1) % n_seeds]:
                    edges.append(
                        [idx, get_node_index(i + 1, j, num_nodes)])

            # Diagonal connections at Voronoi boundaries
            if i < num_nodes - 1 and j < num_nodes - 1:
                diag_distances = [
                    np.sqrt((i + 1 - s[0]) ** 2 + (j + 1 - s[1]) ** 2) for
                    s in seeds]
                diag_closest = np.argmin(diag_distances)

                if diag_closest != closest_seed:
                    edges.append(
                        [idx, get_node_index(i + 1, j + 1, num_nodes)])

    return edges


def generate_circular_brace(num_nodes):
    """Circular/ring bracing pattern"""
    edges = []
    center = (num_nodes - 1) / 2.0

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Distance from center
            r = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            ring = int(r + 0.5)  # Which ring this node belongs to

            # Circumferential connections (along rings)
            if j < num_nodes - 1:
                r_right = np.sqrt(
                    (i - center) ** 2 + (j + 1 - center) ** 2)
                if abs(r - r_right) < 0.7:  # Same ring
                    edges.append(
                        [idx, get_node_index(i, j + 1, num_nodes)])

            if i < num_nodes - 1:
                r_down = np.sqrt(
                    (i + 1 - center) ** 2 + (j - center) ** 2)
                if abs(r - r_down) < 0.7:  # Same ring
                    edges.append(
                        [idx, get_node_index(i + 1, j, num_nodes)])

            # Radial connections (between rings)
            if i < num_nodes - 1 and j < num_nodes - 1:
                r_diag = np.sqrt(
                    (i + 1 - center) ** 2 + (j + 1 - center) ** 2)
                if r_diag > r:  # Pointing outward
                    edges.append(
                        [idx, get_node_index(i + 1, j + 1, num_nodes)])

    return edges


def generate_kagome_pattern(num_nodes):
    """Kagome lattice - triangular with hexagonal voids"""
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            idx = get_node_index(i, j, num_nodes)

            # Create pattern with periodic voids
            if (i + j) % 3 != 0:  # Skip some connections to create voids
                if j < num_nodes - 1:
                    edges.append(
                        [idx, get_node_index(i, j + 1, num_nodes)])
                if i < num_nodes - 1:
                    edges.append(
                        [idx, get_node_index(i + 1, j, num_nodes)])

            # Selective diagonals
            if i < num_nodes - 1 and j < num_nodes - 1:
                if (i * 2 + j) % 3 == 0:
                    edges.append(
                        [idx, get_node_index(i + 1, j + 1, num_nodes)])
                elif (i + j * 2) % 3 == 0:
                    edges.append([get_node_index(i, j + 1, num_nodes),
                                  get_node_index(i + 1, j, num_nodes)])

    return edges

