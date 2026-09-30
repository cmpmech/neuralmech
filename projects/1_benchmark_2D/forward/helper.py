import argparse
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp
import torch

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "../bcs"))  # setups and pixel mesh, shared with optimization/

from boundary import EDGES, load, shape_name
from pixelmesh import PixelMesh, lengths

DATA_DIR = (BASE_DIR / "../../../data/2D_benchmark").resolve()
RGB_PDF_DIR = (BASE_DIR / "../../../results/rgb_pdf").resolve()

with open(BASE_DIR / "../settings.toml", "rb") as f:
    SETTINGS = tomllib.load(f)

OVERLAY = 0.35  # opacity of the black overlay marking the weak phase in plots


def benchmark_parser(setup):
    """arguments shared by all forward drivers; drivers add their material ranges."""
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


def voxel_field(values):
    """piecewise constant mlhp field over the domain, one value per pixel."""
    data = mlhp.DoubleVector(np.ascontiguousarray(values, dtype=float).ravel().tolist())
    return mlhp.scalarFieldFromVoxelData(data, list(values.shape), lengths(values.shape))


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
