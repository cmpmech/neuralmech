import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
from pyevtk.hl import imageToVTK
from scipy import ndimage

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
VOXEL_DIR = DATA_DIR / "geometry/voxel"
RESULTS_DIR = (BASE_DIR / "../../results/abc/geometry").resolve()
DB_PATH = DATA_DIR / "metadata.db"

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
args = parser.parse_args()

# --------------------------------- voxelization settings -----------------------------
STL_IDX = args.geometry  # starts at 1
STL_NAME = f"{STL_IDX:09}_abc"

N = 256  # 128  # 16  # 256
PADDING = 0  # 1  # empty-voxel margin per side so the solid never touches a face
MULTIPLE = 16  # round each side length up to a multiple of this (needs to be consistent with N)

POSTPROCESSING = True

# --------------------------------- load & orient mesh --------------------------------
surface = mlhp.readStl(str(STL_DIR / f"{STL_NAME}.stl"))
(x0, y0, z0), (x1, y1, z1) = surface.boundingBox()
extent = np.array([x1 - x0, y1 - y0, z1 - z0])

# -------------------------------------- voxelize -------------------------------------
axis = int(np.argmax(extent))  # maximum length as highest resolution axis
s = extent[axis] / (
    N - 2 * PADDING
)  # leave room for padding so the longest axis lands on N
counts = [int(np.ceil(e / s)) + 2 * PADDING for e in extent]
ncells = [((n + MULTIPLE - 1) // MULTIPLE) * MULTIPLE for n in counts]
lengths = [n * s for n in ncells]
origin = [bmin - 0.5 * (L - e) for bmin, L, e in zip((x0, y0, z0), lengths, extent)]

centers = [o + s * (np.arange(n) + 0.5) for o, n in zip(origin, ncells)]
X, Y, Z = np.meshgrid(*centers, indexing="ij")

print(X.shape)

domain = mlhp.rayIntersectionDomain(surface)
inside = domain.asfield(0.0, 1.0)
values = np.array(inside(X.ravel(), Y.ravel(), Z.ravel()))
indicator = np.where(values.reshape(ncells) >= 0.5, 255, 0).astype(np.uint8)

Lx, Ly, Lz = lengths

# --------------------------------- connectivity check --------------------------------
structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
labels, ncomponents = ndimage.label(indicator >= 128, structure=structure)
sizes = np.bincount(labels.ravel())[1:]  # drop background (label 0)
connected = ncomponents == 1

# -------------------------------------- logging --------------------------------------
nx, ny, nz = indicator.shape
voxels_solid = int((indicator >= 128).sum())
voxels_total = int(indicator.size)

connection = sqlite3.connect(DB_PATH, timeout=60.0)
connection.execute("PRAGMA journal_mode=WAL")
connection.execute(
    "INSERT INTO geometry(STL_ID, CONNECTED, VOXELS_SOLID, VOXELS_TOTAL,"
    " VOXELS_NX, VOXELS_NY, VOXELS_NZ) VALUES(?, ?, ?, ?, ?, ?, ?)"
    " ON CONFLICT(STL_ID) DO UPDATE SET CONNECTED=excluded.CONNECTED,"
    " VOXELS_SOLID=excluded.VOXELS_SOLID, VOXELS_TOTAL=excluded.VOXELS_TOTAL,"
    " VOXELS_NX=excluded.VOXELS_NX, VOXELS_NY=excluded.VOXELS_NY,"
    " VOXELS_NZ=excluded.VOXELS_NZ",
    (STL_IDX, int(connected), voxels_solid, voxels_total, nx, ny, nz),
)
connection.commit()
connection.close()

# --------------------------------------- export --------------------------------------
if connected:
    VOXEL_DIR.mkdir(parents=True, exist_ok=True)
    out = VOXEL_DIR / f"{STL_NAME}.npz"
    # origin locates the grid in space (= solve's origin_v); spacing is derivable as Lx/Nx
    np.savez(out, indicator=indicator, origin=np.array(origin), Lx=Lx, Ly=Ly, Lz=Lz)
    print(
        f"\tsaved {out}\n\tshape={indicator.shape}\n\tvoxels={np.prod(indicator.shape):.2e}"
    )

    if POSTPROCESSING:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        vti = RESULTS_DIR / STL_NAME
        imageToVTK(
            str(vti),
            origin=tuple(
                float(o) for o in origin
            ),  # plain floats: np.float64 str() breaks the .vti
            spacing=(float(s),) * 3,
            cellData={"indicator": indicator},
        )
        print(f"\tsaved {vti}.vti")
else:
    print("\tskipped export (disconnected)")
