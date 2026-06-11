import argparse
from pathlib import Path

import mlhp
import numpy as np
import wildmeshing as wm

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
STL_DIR = DATA_DIR / "geometry/stl"
MESH_DIR = DATA_DIR / "geometry/mesh"
RESULTS_DIR = (BASE_DIR / "../../results/abc/geometry").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=int, required=True)
args = parser.parse_args()

# ---------------------------------- meshing settings ---------------------------------
STL_IDX = args.geometry  # starts at 1
STL_NAME = f"{STL_IDX:09}_abc"

NELEMENTS = 40  # target tets along the longest bounding-box axis
# NELEMENTS = 60  # target tets along the longest bounding-box axis

POSTPROCESSING = True

# ---------------------------------- tetrahedralize -----------------------------------
# fTetWild (wildmeshing) tetrahedralizes the raw STL directly into a conforming,
# body-fitted interior mesh. It is robust to the ABC STL defects (self-intersections,
# overlapping facets, non-watertight) that make a direct gmsh mesh fail, and needs no
# voxelization. get_tet_mesh() returns interior tets only.
stl = str(STL_DIR / f"{STL_NAME}.stl")
surface = mlhp.readStl(stl)
lo, hi = np.asarray(surface.boundingBox())
extent = hi - lo
edge_length_r = (extent.max() / NELEMENTS) / np.linalg.norm(extent)  # fTetWild knob

tetra = wm.Tetrahedralizer(stop_quality=10, edge_length_r=edge_length_r)
tetra.load_mesh(stl)
tetra.tetrahedralize()
V, T, _ = tetra.get_tet_mesh()

vertices = np.asarray(V, dtype=np.float64)
tets = np.asarray(T, dtype=np.int64)
cells = tets.ravel()  # flat connectivity, 0-based
offsets = np.arange(0, 4 * len(tets) + 1, 4)

# --------------------------------------- export -------------------------------------
MESH_DIR.mkdir(parents=True, exist_ok=True)
out = MESH_DIR / f"{STL_NAME}.npz"
np.savez(out, vertices=vertices, cells=cells, offsets=offsets)
print(f"\tsaved {out}\n\tvertices={len(vertices)}\n\ttets={len(tets)}")

if POSTPROCESSING:
    # round-trip through mlhp (validates the arrays) and write a ParaView .vtu
    mesh = mlhp.makeUnstructuredMesh(vertices, cells, offsets)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vtu = RESULTS_DIR / f"{STL_NAME}_mesh.vtu"
    mesh.writeVtu(str(vtu))
    print(f"\tsaved {vtu}")
