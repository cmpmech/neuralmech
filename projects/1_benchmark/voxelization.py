from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
from pyevtk.hl import imageToVTK
from scipy import ndimage

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data/abc"
STL_DIR = DATA_DIR / "geometry/stl"
VOXEL_DIR = DATA_DIR / "geometry/voxel"
VTI_DIR = BASE_DIR / "../../results/abc/geometry"

# --------------------------------- voxelization settings -----------------------------
# STL_IDX = 1  # starts at 1
for STL_IDX in [3, 4, 5, 13, 14, 15]:
    STL_NAME = f"{STL_IDX:09}_abc"

    # STL_NAME = "Part Studio 1 - Part 1"  # TODO remove when downloaded
    # STL_NAME = "Part Studio 1 - Part 2"  # TODO remove when downloaded

    # resolution is fixed along one reference axis (0=x, 1=y, 2=z, or "max" for the
    # longest axis); voxels are cubic, so the remaining axis counts follow from the
    # (rotated) bounding box
    AXIS = "max"
    N = 256
    # N = 512
    # N = 700
    # optional rigid rotation (degrees) applied around x, y, z before voxelizing
    ROTATION = (0.0, 0.0, 0.0)

    # empty voxel margin on every side, so boundary voxels are unambiguously outside
    PADDING = 2

    EXPORT_VTU = True
    PREVIEW = False

    # verify the voxelized solid is a single connected (6-connected) body; failing
    # filenames are appended to FAILURE_LOG, one per line, for later parsing
    FAILURE_LOG = DATA_DIR / "disconnected.txt"

# --------------------------------- load & orient mesh ---------------------------------
    surface = mlhp.readStl(str(STL_DIR / f"{STL_NAME}.stl"))

    rx, ry, rz = np.deg2rad(ROTATION)
    if any(ROTATION):
        rotation = mlhp.concatenate(
            [
                mlhp.rotation([1.0, 0.0, 0.0], rx),
                mlhp.rotation([0.0, 1.0, 0.0], ry),
                mlhp.rotation([0.0, 0.0, 1.0], rz),
            ]
        )
        surface.transform(rotation)

    (x0, y0, z0), (x1, y1, z1) = surface.boundingBox()
    extent = np.array([x1 - x0, y1 - y0, z1 - z0])

# ------------------------------------- voxelize -------------------------------------
    axis = int(np.argmax(extent)) if AXIS == "max" else AXIS
    s = extent[axis] / N
    ncells = [int(np.ceil(e / s)) + 2 * PADDING for e in extent]
    lengths = [n * s for n in ncells]
    origin = [bmin - 0.5 * (L - e) for bmin, L, e in zip((x0, y0, z0), lengths, extent)]

    centers = [o + s * (np.arange(n) + 0.5) for o, n in zip(origin, ncells)]
    X, Y, Z = np.meshgrid(*centers, indexing="ij")

    domain = mlhp.rayIntersectionDomain(surface)
    inside = domain.asfield(0.0, 1.0)
    values = np.array(inside(X.ravel(), Y.ravel(), Z.ravel()))
    indicator = np.where(values.reshape(ncells) >= 0.5, 255, 0).astype(np.uint8)

    Lx, Ly, Lz = lengths

# ----------------------------- connectivity check -----------------------------
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels, ncomponents = ndimage.label(indicator >= 128, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]  # drop background (label 0)
    connected = ncomponents == 1
    if connected:
        print(f"\tconnected: 1 component, {int(sizes[0])} voxels")
    else:
        total = int(sizes.sum())
        frac = sizes.max() / total if total else 0.0
        print(
            f"\tDISCONNECTED: {ncomponents} components"
            f" (sizes={sorted(sizes.tolist(), reverse=True)},"
            f" largest={frac:.3%})"
        )
        existing = set()
        if FAILURE_LOG.exists():
            existing = set(FAILURE_LOG.read_text().split())
        if STL_NAME not in existing:
            with FAILURE_LOG.open("a") as f:
                f.write(f"{STL_NAME}\n")
        print(f"\tlogged to {FAILURE_LOG}")

# --------------------------------------- export ---------------------------------------
    if connected:
        VOXEL_DIR.mkdir(parents=True, exist_ok=True)
        out = VOXEL_DIR / f"{STL_NAME}.npz"
        np.savez(out, indicator=indicator, Lx=Lx, Ly=Ly, Lz=Lz)
        print(
            f"\tsaved {out}\n\tshape={indicator.shape}\n\tvoxels={np.prod(indicator.shape):.2e}"
        )

        if EXPORT_VTU:
            VTI_DIR.mkdir(parents=True, exist_ok=True)
            vti = VTI_DIR / STL_NAME
            imageToVTK(
                str(vti),
                origin=tuple(origin),
                spacing=(s, s, s),
                cellData={"indicator": indicator},
            )
            print(f"\tsaved {vti}.vti")
    else:
        print("\tskipped export (disconnected)")

# --------------------------------------- preview ---------------------------------------
    if PREVIEW:
        nx, ny, nz = indicator.shape
        slices = [
            (indicator[nx // 2, :, :], "x", "y", "z"),
            (indicator[:, ny // 2, :], "y", "x", "z"),
            (indicator[:, :, nz // 2], "z", "x", "y"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (sl, normal, h, v) in zip(axes, slices):
            ax.imshow(sl.T, origin="lower", cmap="binary", aspect="equal")
            ax.set_title(f"mid-{normal} slice")
            ax.set_xlabel(h)
            ax.set_ylabel(v)
        fig.tight_layout()
        plt.show()
