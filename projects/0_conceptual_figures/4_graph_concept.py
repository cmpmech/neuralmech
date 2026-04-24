import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import qmc

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# ---------------------------- generate graph ----------------------------
sampler = qmc.LatinHypercube(d=2, seed=0)
points = sampler.random(n=26)
tri = Delaunay(points)

if not args.book:
    fig, ax = plt.subplots()
    ax.triplot(points[:, 0], points[:, 1], tri.simplices, lw=1.2)
    ax.scatter(points[:, 0], points[:, 1], color="k")
    plt.show()

# ---------------------------- extract edges -----------------------------
edges = set()
for tri_nodes in tri.simplices:
    i, j, k = tri_nodes
    edges.add(tuple(sorted((i, j))))
    edges.add(tuple(sorted((j, k))))
    edges.add(tuple(sorted((k, i))))
edges = np.array(list(edges))

# --------------------------- receptive field ----------------------------
point0 = np.expand_dims(points[21], 0)
point1 = points[[3, 22, 19, 23, 17]]
point2 = points[[18, 10, 0, 14, 5, 11, 2, 1, 7, 20, 15]]
edges12 = [
    points[[3, 18]],
    points[[3, 15]],
    points[[3, 20]],
    points[[3, 17]],
    points[[3, 22]],
    points[[22, 18]],
    points[[22, 10]],
    points[[22, 0]],
    points[[22, 14]],
    points[[22, 19]],
    points[[19, 14]],
    points[[19, 5]],
    points[[19, 23]],
    points[[23, 5]],
    points[[23, 11]],
    points[[23, 2]],
    points[[23, 1]],
    points[[23, 17]],
    points[[17, 1]],
    points[[17, 7]],
    points[[17, 20]],
]

if not args.book:
    fig, ax = plt.subplots()
    ax.triplot(points[:, 0], points[:, 1], tri.simplices, lw=1.2)
    ax.scatter(points[:, 0], points[:, 1], color="k")
    for i, (x, y) in enumerate(points):
        ax.text(x, y, i, ha="left", va="top", fontsize=8)
    ax.plot(point0[:, 0], point0[:, 1], "ro")
    ax.scatter(point1[:, 0], point1[:, 1], color="b")
    ax.scatter(point2[:, 0], point2[:, 1], color="y")
    plt.show()

if args.book:
    np.savetxt(RESULTS_DIR / "graph_concept_nodes.txt", points, fmt="%.6f %.6f")
    np.savetxt(RESULTS_DIR / "graph_concept_edges.txt", edges, fmt="%d %d")
    np.savetxt(RESULTS_DIR / "graph_concept_nodes0.txt", point0, fmt="%.6f %.6f")
    np.savetxt(RESULTS_DIR / "graph_concept_nodes1.txt", point1, fmt="%.6f %.6f")
    np.savetxt(RESULTS_DIR / "graph_concept_nodes2.txt", point2, fmt="%.6f %.6f")
    np.savetxt(
        RESULTS_DIR / "graph_concept_edges12_a.txt",
        np.array(edges12)[:, 0],
        fmt="%.6f %.6f",
    )
    np.savetxt(
        RESULTS_DIR / "graph_concept_edges12_b.txt",
        np.array(edges12)[:, 1],
        fmt="%.6f %.6f",
    )
