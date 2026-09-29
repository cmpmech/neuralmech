import argparse
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp
import torch

from bcs.boundary import OFFSET, load, shape_name

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../data/2D_benchmark").resolve()
RGB_PDF_DIR = (BASE_DIR / "../../../results/rgb_pdf").resolve()

with open(BASE_DIR / "../settings.toml", "rb") as f:
    SETTINGS = tomllib.load(f)

OVERLAY = 0.35  # opacity of the black overlay marking the weak phase in plots


def benchmark_parser(setup):
    """arguments shared by all physics drivers; drivers add their material ranges."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ctscans")
    parser.add_argument("--type", default="B_HAI")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256, help="pixels along x_1")
    parser.add_argument("--aspect", type=int, default=1, help="long : short side")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="binarize the gray value scaled to [0, 1]; grayscale if omitted",
    )
    parser.add_argument(
        "--setup",
        default=setup,
        help="boundary conditions and sources from data/2D_benchmark/setups",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--book", action="store_true")
    return parser


def load_phase(args):
    """phase indicator g in [0, 1] per pixel, g[i, j] at pixel (x_i, y_j)."""
    shape = shape_name(args.resolution, args.aspect)
    file = DATA_DIR / "geometries" / args.source / f"{args.type}_{shape}.pt"
    image = torch.load(file, weights_only=False, map_location="cpu")[
        args.sample
    ].numpy()
    g = image / 255.0
    return (g >= args.threshold).astype(float) if args.threshold is not None else g


def load_setup(args):
    """(BoundaryConditions or None, list of source dicts)."""
    shape = shape_name(args.resolution, args.aspect)
    return load(DATA_DIR / "setups" / f"{args.setup}_{shape}.pt")


def interpolate(g, low, high):
    return low + g * (high - low)


def lengths(shape):
    """side lengths of a domain of `shape` pixels, the long side x_1 of unit length."""
    return [1.0, shape[1] / shape[0]]


def voxel_field(values):
    """piecewise constant mlhp field over the domain, one value per pixel."""
    data = mlhp.DoubleVector(np.ascontiguousarray(values, dtype=float).ravel().tolist())
    return mlhp.scalarFieldFromVoxelData(data, list(values.shape), lengths(values.shape))


class PixelMesh:
    """one bilinear element per square pixel, so the dofs sit on the pixel nodes.

    `dof[i, j, c]` is the dof of component c at node (x_i, y_j).
    """

    def __init__(self, resolution, nfields, aspect=1):
        self.R, self.nfields = resolution, nfields
        self.shape = (resolution, resolution // aspect)
        self.grid = mlhp.makeRefinedGrid(mlhp.makeGrid(list(self.shape), lengths(self.shape)))
        self.basis = mlhp.makeHpTensorSpace(self.grid, degree=1, nfields=nfields)
        self.ndof = self.basis.ndof()

        # linear functions are reproduced exactly, so projecting x and y yields the nodes
        ij = []
        for coordinate in ("x", "y"):
            field = mlhp.scalarField(2, coordinate)
            if nfields > 1:
                field = mlhp.vectorField(2, [field] * nfields)
            ij.append(
                np.rint(np.array(mlhp.projectOnto(self.basis, field)) * resolution)
            )
        component = np.array(self.basis.fieldmap()) if nfields > 1 else 0
        self.dof = np.zeros((self.shape[0] + 1, self.shape[1] + 1, nfields), dtype=int)
        self.dof[ij[0].astype(int), ij[1].astype(int), component] = np.arange(self.ndof)

    def closed_edge_dofs(self, edge):
        """dofs (nodes, nfields) of all nodes along edge 0-3 (left, right, bottom, top)."""
        return [self.dof[0], self.dof[-1], self.dof[:, 0], self.dof[:, -1]][edge]

    def edge_dofs(self, edge):
        """dofs of the nodes edge 0-3 holds: all but the corner it passes on."""
        dofs = self.closed_edge_dofs(edge)
        return dofs[1:] if OFFSET[edge] else dofs[:-1]

    def dirichlet(self, bc, scale=1.0):
        """mlhp dirichlet dofs [indices, values], prescribed values scaled by `scale`."""
        ids, values = [], []
        for edge in range(4):
            dofs = self.edge_dofs(edge)
            mask = bc.dirichlet[edge, : len(dofs), : self.nfields].numpy()
            ids.append(dofs[mask])
            values.append(scale * bc.u[edge, : len(dofs), : self.nfields].numpy()[mask])
        ids, first = np.unique(np.concatenate(ids), return_index=True)
        return [ids.tolist(), np.concatenate(values)[first].tolist()]

    def neumann(self, bc, scale=1.0):
        """nodal loads from the edge tractions, lumped with the trapezoidal rule."""
        h = 1.0 / self.R
        load = np.zeros(self.ndof)
        for edge in range(4):
            dofs = self.closed_edge_dofs(edge)
            weights = np.full(len(dofs), h)
            weights[[0, -1]] = 0.5 * h
            t = (bc.t * bc.neumann)[edge, : len(dofs) - 1, : self.nfields].numpy()
            # the traction reaches the corner the edge passes on, continued from its neighbor
            t = np.concatenate([t[:1], t]) if OFFSET[edge] else np.concatenate([t, t[-1:]])
            np.add.at(load, dofs.ravel(), (weights[:, None] * t).ravel())
        return scale * load

    def sources(self, sources, g, scale=1.0):
        """nodal loads of point, gaussian and gravity sources (nodal quadrature).

        Positions are fractions of each side. Gravity scales with the phase g averaged onto
        the nodes (density ~ gray value).
        """
        (R1, R2), h = self.shape, 1.0 / self.R
        x, y = np.meshgrid(np.arange(R1 + 1) * h, np.arange(R2 + 1) * h, indexing="ij")
        area = np.full((R1 + 1, R2 + 1), h * h)
        area[[0, -1]] *= 0.5
        area[:, [0, -1]] *= 0.5
        load = np.zeros(self.ndof)
        for s in sources:
            amplitude = np.array(s["amplitude"])[: self.nfields]
            if s["kind"] == "point":
                i, j = (int(round(p * n)) for p, n in zip(s["position"], self.shape))
                load[self.dof[i, j]] += amplitude
                continue
            if s["kind"] == "gaussian":
                w = s["width"]
                x0, y0 = (p * n * h for p, n in zip(s["position"], self.shape))
                r2 = (x - x0) ** 2 + (y - y0) ** 2
                density = np.exp(-0.5 * r2 / w**2) / (2 * np.pi * w**2)
            else:
                padded = np.pad(g, 1, mode="edge")
                density = 0.25 * (
                    padded[1:, 1:]
                    + padded[:-1, 1:]
                    + padded[1:, :-1]
                    + padded[:-1, :-1]
                )
            np.add.at(
                load,
                self.dof.reshape(-1, self.nfields),
                (density * area).reshape(-1, 1) * amplitude,
            )
        return scale * load

    def cell_values(self, function):
        """(R_1, R_2, ncomponents) values of an mlhp mesh function at the pixel centres."""
        ncells = self.shape[0] * self.shape[1]
        values = np.array([function(cell, [0.0, 0.0]) for cell in range(ncells)])
        if not hasattr(self, "pixel"):
            xy = mlhp.vectorField(
                2, [mlhp.scalarField(2, "x"), mlhp.scalarField(2, "y")]
            )
            centres = mlhp.meshFunction(self.grid, xy)
            ij = np.array([centres(cell, [0.0, 0.0]) for cell in range(ncells)])
            self.pixel = tuple(np.floor(ij * self.R).astype(int).T)
        out = np.zeros((*self.shape, values.shape[1]))
        out[self.pixel] = values
        return out

    @staticmethod
    def pixel_mean(nodal):
        """(R_1, R_2, ...) pixel averages of nodal values (R_1 + 1, R_2 + 1, ...)."""
        return 0.25 * (
            nodal[1:, 1:] + nodal[:-1, 1:] + nodal[1:, :-1] + nodal[:-1, :-1]
        )

    def internal(self, load, dirichlet):
        """restrict a full-length vector to the free dofs (mlhp's reduced ordering)."""
        return np.delete(load, dirichlet[0])

    def nodal(self, dofs):
        """(R_1 + 1, R_2 + 1, nfields) nodal values of a full dof vector."""
        return np.asarray(dofs)[self.dof]


def csr(matrix):
    """scipy view of an assembled mlhp matrix."""
    return sp.csr_matrix(
        (
            np.asarray(matrix.data_array),
            np.asarray(matrix.indices_array),
            np.asarray(matrix.indptr_array),
        ),
        shape=tuple(matrix.shape),
    )


def solve(matrix, vector):
    """direct solve of an assembled mlhp system (robust to high material contrast)."""
    return pypardiso.spsolve(csr(matrix), np.asarray(vector))


def pixel_gradient(nodal):
    """gradient at the pixel centres, (R_1, R_2, nfields, 2), from (R_1 + 1, R_2 + 1, nfields)."""
    R = nodal.shape[0] - 1  # square pixels of size 1 / R_1
    dx = 0.5 * R * (nodal[1:, 1:] - nodal[:-1, 1:] + nodal[1:, :-1] - nodal[:-1, :-1])
    dy = 0.5 * R * (nodal[1:, 1:] - nodal[1:, :-1] + nodal[:-1, 1:] - nodal[:-1, :-1])
    return np.stack([dx, dy], axis=-1)


def save_solution(args, physics, **fields):
    shape = shape_name(args.resolution, args.aspect)
    name = f"{args.source}_{args.type}_{args.sample}_{shape}_{args.setup}.pt"
    file = DATA_DIR / "solutions" / physics / name
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: torch.as_tensor(value) for key, value in fields.items()}, file)
    print(f"saved {file}")


def plot_field(field, g, cmap, name, book, vmin=None, vmax=None, solid=None):
    """field (nodal or per pixel) over the domain, weak phase darkened.

    With a `solid` gray level, the weak phase (g < 0.5) is painted opaque instead, for
    fields that do not exist there (fluid flow).
    """
    width, height = lengths(g.shape)
    fig, ax = plt.subplots(figsize=(5 * width, 5 * height), dpi=100 if book else 150)
    extent = (0.0, width, 0.0, height)
    ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    shade = np.zeros((*g.T.shape, 4))
    shade[..., 3] = OVERLAY * (1.0 - g.T)
    if solid is not None:
        shade[..., :3] = solid
        shade[..., 3] = g.T < 0.5
    ax.imshow(shade, origin="lower", extent=extent, interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if book:
        fig.savefig(RGB_PDF_DIR / f"{name}.pdf")
        plt.close(fig)
    else:
        plt.show()
