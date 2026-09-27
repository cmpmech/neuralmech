import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp
import torch

from bcs.boundary import load

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../data/2D_benchmark").resolve()
RGB_PDF_DIR = (BASE_DIR / "../../../results/rgb_pdf").resolve()

OVERLAY = 0.35  # opacity of the black overlay marking the weak phase in plots


def benchmark_parser(setup):
    """arguments shared by all physics drivers; drivers add their material ranges."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ctscans")
    parser.add_argument("--type", default="B_HAI")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
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
    file = DATA_DIR / "geometries" / args.source / f"{args.type}_{args.resolution}.pt"
    image = torch.load(file, weights_only=False, map_location="cpu")[
        args.sample
    ].numpy()
    g = image / 255.0
    return (g >= args.threshold).astype(float) if args.threshold is not None else g


def load_setup(args):
    """(BoundaryConditions or None, list of source dicts)."""
    return load(DATA_DIR / "setups" / f"{args.setup}_{args.resolution}.pt")


def interpolate(g, low, high):
    return low + g * (high - low)


def voxel_field(values):
    """piecewise constant mlhp field over the unit square, one value per pixel."""
    data = mlhp.DoubleVector(np.ascontiguousarray(values, dtype=float).ravel().tolist())
    return mlhp.scalarFieldFromVoxelData(data, list(values.shape), [1.0, 1.0])


class PixelMesh:
    """one bilinear element per pixel, so the dofs sit on the pixel nodes.

    `dof[i, j, c]` is the dof of component c at node (x_i, y_j).
    """

    def __init__(self, resolution, nfields):
        self.R, self.nfields = resolution, nfields
        self.grid = mlhp.makeRefinedGrid(mlhp.makeGrid([resolution] * 2, [1.0, 1.0]))
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
        self.dof = np.zeros((resolution + 1, resolution + 1, nfields), dtype=int)
        self.dof[ij[0].astype(int), ij[1].astype(int), component] = np.arange(self.ndof)

    def edge_dofs(self, edge):
        """dofs (R + 1, nfields) along edge 0-3 (left, right, bottom, top)."""
        return [self.dof[0], self.dof[-1], self.dof[:, 0], self.dof[:, -1]][edge]

    def dirichlet(self, bc, scale=1.0):
        """mlhp dirichlet dofs [indices, values], prescribed values scaled by `scale`."""
        ids, values = [], []
        for edge in range(4):
            mask = bc.dirichlet[edge, :, : self.nfields].numpy()
            ids.append(self.edge_dofs(edge)[mask])
            values.append(scale * bc.u[edge, :, : self.nfields].numpy()[mask])
        ids, first = np.unique(np.concatenate(ids), return_index=True)
        return [ids.tolist(), np.concatenate(values)[first].tolist()]

    def neumann(self, bc, scale=1.0):
        """nodal loads from the edge tractions, lumped with the trapezoidal rule."""
        h = 1.0 / self.R
        weights = np.full(self.R + 1, h)
        weights[[0, -1]] = 0.5 * h
        load = np.zeros(self.ndof)
        for edge in range(4):
            t = (bc.t * bc.neumann)[edge, :, : self.nfields].numpy()
            np.add.at(
                load, self.edge_dofs(edge).ravel(), (weights[:, None] * t).ravel()
            )
        return scale * load

    def sources(self, sources, g, scale=1.0):
        """nodal loads of point, gaussian and gravity sources (nodal quadrature).

        Gravity scales with the phase g averaged onto the nodes (density ~ gray value).
        """
        R, h = self.R, 1.0 / self.R
        x, y = np.meshgrid(
            np.linspace(0, 1, R + 1), np.linspace(0, 1, R + 1), indexing="ij"
        )
        area = np.full((R + 1, R + 1), h * h)
        area[[0, -1]] *= 0.5
        area[:, [0, -1]] *= 0.5
        load = np.zeros(self.ndof)
        for s in sources:
            amplitude = np.array(s["amplitude"])[: self.nfields]
            if s["kind"] == "point":
                i, j = (int(round(p * R)) for p in s["position"])
                load[self.dof[i, j]] += amplitude
                continue
            if s["kind"] == "gaussian":
                w = s["width"]
                r2 = (x - s["position"][0]) ** 2 + (y - s["position"][1]) ** 2
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
        """(R, R, ncomponents) values of an mlhp mesh function at the pixel centres."""
        values = np.array([function(cell, [0.0, 0.0]) for cell in range(self.R**2)])
        if not hasattr(self, "pixel"):
            xy = mlhp.vectorField(
                2, [mlhp.scalarField(2, "x"), mlhp.scalarField(2, "y")]
            )
            centres = mlhp.meshFunction(self.grid, xy)
            ij = np.array([centres(cell, [0.0, 0.0]) for cell in range(self.R**2)])
            self.pixel = tuple(np.floor(ij * self.R).astype(int).T)
        out = np.zeros((self.R, self.R, values.shape[1]))
        out[self.pixel] = values
        return out

    @staticmethod
    def pixel_mean(nodal):
        """(R, R, ...) pixel averages of nodal values (R + 1, R + 1, ...)."""
        return 0.25 * (
            nodal[1:, 1:] + nodal[:-1, 1:] + nodal[1:, :-1] + nodal[:-1, :-1]
        )

    def internal(self, load, dirichlet):
        """restrict a full-length vector to the free dofs (mlhp's reduced ordering)."""
        return np.delete(load, dirichlet[0])

    def nodal(self, dofs):
        """(R + 1, R + 1, nfields) nodal values of a full dof vector."""
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
    """gradient at the pixel centres, (R, R, nfields, 2), from (R + 1, R + 1, nfields)."""
    R = nodal.shape[0] - 1
    dx = 0.5 * R * (nodal[1:, 1:] - nodal[:-1, 1:] + nodal[1:, :-1] - nodal[:-1, :-1])
    dy = 0.5 * R * (nodal[1:, 1:] - nodal[1:, :-1] + nodal[:-1, 1:] - nodal[:-1, :-1])
    return np.stack([dx, dy], axis=-1)


def save_solution(args, physics, **fields):
    name = f"{args.source}_{args.type}_{args.sample}_{args.resolution}_{args.setup}.pt"
    file = DATA_DIR / "solutions" / physics / name
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: torch.as_tensor(value) for key, value in fields.items()}, file)
    print(f"saved {file}")


def plot_field(field, g, cmap, name, book, vmin=None, vmax=None, solid=None):
    """field (nodal or per pixel) over the unit square, weak phase darkened.

    With a `solid` gray level, the weak phase (g < 0.5) is painted opaque instead, for
    fields that do not exist there (fluid flow).
    """
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100 if book else 150)
    extent = (0.0, 1.0, 0.0, 1.0)
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
