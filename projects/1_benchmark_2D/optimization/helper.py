import argparse
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import torch
from tqdm import tqdm

from solvers.optimization import DensityFilter, StructuredFEM, dsimp, simp

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "../bcs"))  # setups and pixel mesh, shared with forward/

from boundary import load, shape_name
from pixelmesh import PixelMesh, lengths

DATA_DIR = (BASE_DIR / "../../../data/2D_benchmark").resolve()
RGB_PDF_DIR = (BASE_DIR / "../../../results/rgb_pdf").resolve()

with open(BASE_DIR / "../settings.toml", "rb") as f:
    SETTINGS = tomllib.load(f)

OPTIMIZATION = SETTINGS["optimization"]
FRAMES = [1, 2, 4, 8, 16, 32, 64]  # iterates in the book figure


def optimization_parser(setup):
    """arguments shared by all optimization drivers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=256, help="pixels along x_1")
    parser.add_argument("--aspect", type=int, default=1, help="long : short side")
    parser.add_argument(
        "--setup",
        default=setup,
        help="boundary conditions and sources from data/2D_benchmark/setups",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--book", action="store_true")
    return parser


def load_setup(args):
    """(BoundaryConditions, list of source dicts)."""
    shape = shape_name(args.resolution, args.aspect)
    return load(DATA_DIR / "setups" / f"{args.setup}_{shape}.pt")


def element_matrices(mesh, integrand):
    """unit-material matrix of one pixel element, (1, ndof_e, ndof_e)."""
    h = 1.0 / mesh.R
    grid = mlhp.makeRefinedGrid(mlhp.makeGrid([1, 1], [h, h]))
    basis = mlhp.makeHpTensorSpace(grid, degree=1, nfields=mesh.nfields)
    return mlhp.integratePartitionMatrices(
        basis,
        integrand,
        mlhp.gridQuadrature(nsubcells=[1, 1]),
        mlhp.absoluteQuadratureOrder([2, 2]),
    )


def optimize(mesh, integrand, bc, force, volume, low, high):
    """minimum compliance design by SIMP, a density filter and optimality criteria.

    Returns the physical (filtered) designs of all iterates, (iterations + 1, R_1, R_2),
    the first the uniform initial design, and the compliance of each.
    """
    fixed, values = mesh.dirichlet(bc)
    assert not np.any(values), "compliance needs homogeneous dirichlet conditions"
    free = np.setdiff1d(np.arange(mesh.ndof), fixed)
    efts = np.array(mesh.basis.locationMaps())
    fem = StructuredFEM(
        efts, free, mesh.ndof, element_matrices(mesh, integrand), mesh.shape
    )
    density_filter = DensityFilter(OPTIMIZATION["filter_radius"], mesh.shape)
    penal, move = OPTIMIZATION["penalization"], OPTIMIZATION["move"]

    x = np.full(mesh.shape, volume)
    designs, compliance = [density_filter(x)], []
    for _ in tqdm(range(OPTIMIZATION["iterations"])):
        u = np.zeros(mesh.ndof)
        u[free] = fem.solve(simp(designs[-1], penal, low, high), force[free])
        compliance.append(force @ u)

        dc = -dsimp(designs[-1], penal, low, high) * fem.element_energy(u)
        dc = density_filter.adjoint(dc)
        dv = density_filter.adjoint(np.ones(mesh.shape))

        # bisection on the lagrange multiplier of the volume constraint
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            x_new = x * np.sqrt(np.maximum(0.0, -dc) / (dv * lmid))
            x_new = np.clip(x_new, np.maximum(0.0, x - move), np.minimum(1.0, x + move))
            if density_filter(x_new).mean() > volume:
                l1 = lmid
            else:
                l2 = lmid
        x = x_new
        designs.append(density_filter(x))

    u = np.zeros(mesh.ndof)
    u[free] = fem.solve(simp(designs[-1], penal, low, high), force[free])
    compliance.append(force @ u)
    return np.array(designs), np.array(compliance)


def save_designs(args, problem, designs, compliance):
    """designs as gray values (uint8, like the geometries) and their compliance."""
    shape = shape_name(args.resolution, args.aspect)
    file = DATA_DIR / "optimization" / problem / f"{args.setup}_{shape}.pt"
    file.parent.mkdir(parents=True, exist_ok=True)
    gray = torch.as_tensor(np.rint(255 * designs).astype(np.uint8))
    torch.save({"designs": gray, "compliance": torch.as_tensor(compliance)}, file)
    print(f"saved {file}")


def plot_frames(designs, name, book):
    """the iterates in FRAMES, solid black as usual in topology optimization."""
    width, height = lengths(designs.shape[1:])
    if not book:
        fig, axs = plt.subplots(1, len(FRAMES), figsize=(2 * len(FRAMES), 2.2 * height))
        for ax, k in zip(axs, FRAMES):
            ax.imshow(designs[k].T, cmap="gray_r", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"iteration {k}")
            ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.02)
        plt.show()
        return
    for k in FRAMES:
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(
            designs[k].T,
            cmap="gray_r",
            origin="lower",
            vmin=0,
            vmax=1,
            interpolation="none",
        )
        ax.axis("off")
        fig.savefig(RGB_PDF_DIR / f"{name}_{k}.pdf")
        plt.close(fig)
