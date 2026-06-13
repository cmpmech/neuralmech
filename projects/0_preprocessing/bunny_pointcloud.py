from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
EXT_DATA_DIR = BASE_DIR / "../../external_data"
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results/3D"

# ----------------------------------- settings ----------------------------------------
PLY_FILE = EXT_DATA_DIR / "bunny/reconstruction/bun_zipper.ply"
VOXEL_SIZE = 0.006  # 0.003  # voxel edge length in bunny units (~m); larger = sparser

# ----------------------------------- load ply ----------------------------------------
header = []
with open(PLY_FILE) as f:
    for line in f:
        header.append(line.strip())
        if line.strip() == "end_header":
            break

n_vertices = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
points = np.loadtxt(
    PLY_FILE, skiprows=len(header), max_rows=n_vertices, usecols=(0, 1, 2)
)

# ------------------------------- voxel downsample ------------------------------------
voxel = np.floor(points / VOXEL_SIZE).astype(np.int64)
_, keep = np.unique(voxel, axis=0, return_index=True)  # one point per occupied voxel
cloud = points[np.sort(keep)]

# -------------------------------- export for torch ----------------------------------
npz_out = DATA_DIR / "bunny_pointcloud.npz"
np.savez(npz_out, points=cloud.astype(np.float32), voxel_size=VOXEL_SIZE)

# ------------------------------- export for paraview --------------------------------
m = len(cloud)
lines = [
    "# vtk DataFile Version 3.0",
    "stanford bunny point cloud",
    "ASCII",
    "DATASET POLYDATA",
]
lines.append(f"POINTS {m} float")
lines.extend(f"{x} {y} {z}" for x, y, z in cloud)
lines.append(f"VERTICES {m} {2 * m}")
lines.extend(f"1 {i}" for i in range(m))

vtk_out = RESULTS_DIR / "bunny_pointcloud.vtk"
vtk_out.write_text("\n".join(lines) + "\n")

print(
    f"\t{n_vertices} -> {m} points (voxel={VOXEL_SIZE})\n\tsaved {npz_out}\n\tsaved {vtk_out}"
)
