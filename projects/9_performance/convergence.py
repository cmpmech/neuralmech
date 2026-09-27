import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from matplotlib.tri import Triangulation

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# quarter of a perforated plate under uniaxial tension
DIM = 2

# geometry
LENGTH = 1.0
RADIUS = 0.5  # hole centered at the origin

# physics
E = 206900e6
NU = 0.29
TRACTION = 100e6  # on the right face, in x
REFERENCE_ENERGY = 5.787699646378e4  # overkill, p = 3 on the conforming mesh of level 6

# discretization
LEVELS = {"p1": range(7), "p3": range(6)}  # conforming gmsh meshes of create_meshes.py
P_LEVEL = 0  # p-refinement on the conforming mesh with 2 quadrilaterals
P_ORDERS = range(1, 11)  # gmsh quadrilaterals end at order 10

# postprocessing
SNAPSHOTS = {"p1": [0, 2, 4], "p3": [0, 2, 4], "p": [1, 5, 10]}  # mesh levels, degrees for p
U_RANGE = (-1e-4, 1.9e-3)  # shared color scale of the x displacement in m
SUBDIVISIONS = 8  # plotting points per element edge
MESH_WIDTH = 1.5  # in points of the 3 inch figure, printed at about a fifth

# --------------------------------------- helper --------------------------------------
D_MATRIX = E / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])


def lagrange(order, xi):  # equidistant 1D Lagrange polynomials on [-1, 1] and derivatives
    nodes = np.linspace(-1.0, 1.0, order + 1)
    values = np.ones((len(xi), order + 1))
    derivatives = np.zeros((len(xi), order + 1))
    for a in range(order + 1):
        others = [b for b in range(order + 1) if b != a]
        for b in others:
            values[:, a] *= (xi - nodes[b]) / (nodes[a] - nodes[b])
        for c in others:
            term = np.ones(len(xi)) / (nodes[a] - nodes[c])
            for b in others:
                if b != c:
                    term *= (xi - nodes[b]) / (nodes[a] - nodes[b])
            derivatives[:, a] += term
    return values, derivatives


def shape_functions(order, xi, eta):  # tensor product, xi fastest
    nx, dnx = lagrange(order, xi)
    ny, dny = lagrange(order, eta)
    n = np.einsum("qj,qi->qji", ny, nx).reshape(len(xi), -1)
    dn = np.stack(
        [
            np.einsum("qj,qi->qji", ny, dnx).reshape(len(xi), -1),
            np.einsum("qj,qi->qji", dny, nx).reshape(len(xi), -1),
        ],
        axis=2,
    )
    return n, dn


def solve_conforming(order, level):
    mesh = np.load(DATA_DIR / f"perforated_plate_p{order}_{level}.npz")
    coords, elements = mesh["coords"], mesh["elements"]
    nnodes = len(coords)

    points, weights = np.polynomial.legendre.leggauss(order + 2)
    xi, eta = [g.ravel() for g in np.meshgrid(points, points, indexing="xy")]
    w = np.outer(weights, weights).ravel()
    _, dn = shape_functions(order, xi, eta)

    x = coords[elements]
    jacobian = np.einsum("eai,qak->eqik", x, dn)
    det = np.linalg.det(jacobian)
    assert det.min() > 0.0, "inverted element"
    grads = np.einsum("qak,eqki->eqai", dn, np.linalg.inv(jacobian))

    nelements, nquad, nshape, _ = grads.shape
    b = np.zeros((nelements, nquad, 3, 2 * nshape))
    b[:, :, 0, 0::2] = grads[..., 0]
    b[:, :, 1, 1::2] = grads[..., 1]
    b[:, :, 2, 0::2] = grads[..., 1]
    b[:, :, 2, 1::2] = grads[..., 0]
    ke = np.einsum("eqji,jk,eqkl,eq->eil", b, D_MATRIX, b, det * w)

    dofs = np.stack([2 * elements, 2 * elements + 1], axis=2).reshape(nelements, -1)
    rows = np.repeat(dofs, 2 * nshape, axis=1).ravel()
    cols = np.tile(dofs, (1, 2 * nshape)).ravel()
    stiffness = scipy.sparse.csr_matrix((ke.ravel(), (rows, cols)), shape=(2 * nnodes,) * 2)

    # traction on the element edges lying on the right face
    force = np.zeros(2 * nnodes)
    edge_n, edge_dn = lagrange(order, points)
    grid = elements.reshape(nelements, order + 1, order + 1)
    for edge in [grid[:, :, 0], grid[:, :, -1], grid[:, 0, :], grid[:, -1, :]]:
        on_face = np.all(np.isclose(coords[edge, 0], LENGTH), axis=1)
        for nodes in edge[on_face]:
            ds = np.linalg.norm(edge_dn @ coords[nodes], axis=1) * weights
            np.add.at(force, 2 * nodes, TRACTION * edge_n.T @ ds)

    fixed = np.concatenate(
        [
            2 * np.flatnonzero(np.isclose(coords[:, 0], 0.0)),
            2 * np.flatnonzero(np.isclose(coords[:, 1], 0.0)) + 1,
        ]
    )
    free = np.setdiff1d(np.arange(2 * nnodes), fixed)
    u = np.zeros(2 * nnodes)
    u[free] = scipy.sparse.linalg.spsolve(stiffness[free][:, free], force[free])

    energy = 0.5 * force @ u
    return len(free), energy, (coords, elements, order, u)


def plot_conforming(coords, elements, order, u):
    m = SUBDIVISIONS + 1
    ref = np.linspace(-1.0, 1.0, m)
    xi, eta = [g.ravel() for g in np.meshgrid(ref, ref, indexing="xy")]
    n, _ = shape_functions(order, xi, eta)
    x = np.einsum("qa,eai->eqi", n, coords[elements])
    ux = np.einsum("qa,ea->eq", n, u[2 * elements])

    i, j = np.meshgrid(np.arange(SUBDIVISIONS), np.arange(SUBDIVISIONS), indexing="xy")
    corner = (j * m + i).ravel()
    lower = np.stack([corner, corner + 1, corner + m + 1], axis=1)
    upper = np.stack([corner, corner + m + 1, corner + m], axis=1)
    cells = np.concatenate([lower, upper])
    triangles = (cells[None] + m * m * np.arange(len(elements))[:, None, None]).reshape(-1, 3)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.tricontourf(
        Triangulation(x[..., 0].ravel(), x[..., 1].ravel(), triangles),
        ux.ravel(),
        levels=np.linspace(*U_RANGE, 64),
        cmap="turbo",
        extend="both",
    )
    k = np.arange(m)
    outline = np.concatenate([k, k * m + m - 1, m * m - 1 - k, (m - 1 - k) * m])
    for edges in x[:, outline]:
        ax.plot(edges[:, 0], edges[:, 1], color="white", linewidth=MESH_WIDTH)
    return fig, ax


# ------------------------------------- convergence -----------------------------------
tic = time.time()
studies = {}
for name, levels in LEVELS.items():
    studies[name] = [(level, *solve_conforming(int(name[1]), level)) for level in levels]
studies["p"] = [(order, *solve_conforming(order, P_LEVEL)) for order in P_ORDERS]
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
error_of = lambda energy: math.sqrt(abs(REFERENCE_ENERGY - energy) / REFERENCE_ENERGY)
for name, study in studies.items():
    ndofs = [ndof for _, ndof, _, _ in study]
    errors = [error_of(energy) for _, _, energy, _ in study]
    for (level, ndof, _, state), error in zip(study, errors):
        print(f"{name}  level {level}  dofs {ndof}  error {error:.2e}")
        if level not in SNAPSHOTS[name]:
            continue
        fig, ax = plot_conforming(*state)
        ax.set_xlim(0.0, LENGTH)
        ax.set_ylim(0.0, LENGTH)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if args.book:
            plt.savefig(RGB_PDF_DIR / f"convergence_{name}_{level}.pdf")
            plt.close()

# -------------------------------- book postprocessing --------------------------------
    if args.book:
        save_csv(CSV_DIR / f"convergence_{name}.csv", ndof=ndofs, error=errors)

if not args.book:
    plt.show()
