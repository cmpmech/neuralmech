import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
STL_DIR = (BASE_DIR / "../../data/abc/geometry/stl").resolve()

# ----------------------------------- bar geometry ------------------------------------
# Axis-aligned rectangular bar saved as geometry 0: with no curvature it voxelizes
# without staircasing, so the voxel and FCM geometries are identical and any remaining
# voxel-vs-FCM discrepancy must come from the BCs/load, not the geometry.
GEOMETRY = 0
LENGTHS = (0.2, 0.05, 0.05)  # (Lx, Ly, Lz): elongated along x for an x-tension test
ORIGIN = (0.0, 0.0, 0.0)

name = f"{GEOMETRY:09}_abc"

x0, y0, z0 = ORIGIN
x1, y1, z1 = (ORIGIN[d] + LENGTHS[d] for d in range(3))
v = np.array(
    [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
)

# each face: 4 corner indices wound CCW seen from OUTSIDE, plus its outward unit normal
faces = [
    ([0, 3, 2, 1], (0.0, 0.0, -1.0)),
    ([4, 5, 6, 7], (0.0, 0.0, 1.0)),
    ([0, 1, 5, 4], (0.0, -1.0, 0.0)),
    ([3, 7, 6, 2], (0.0, 1.0, 0.0)),
    ([0, 4, 7, 3], (-1.0, 0.0, 0.0)),
    ([1, 2, 6, 5], (1.0, 0.0, 0.0)),
]

lines = [f"solid {name}"]
for quad, n in faces:
    for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
        lines.append(f"  facet normal {n[0]:.1f} {n[1]:.1f} {n[2]:.1f}")
        lines.append("    outer loop")
        for i in tri:
            lines.append(f"      vertex {v[i, 0]:.9g} {v[i, 1]:.9g} {v[i, 2]:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
lines.append(f"endsolid {name}")

STL_DIR.mkdir(parents=True, exist_ok=True)
out = STL_DIR / f"{name}.stl"
out.write_text("\n".join(lines) + "\n")
print(f"-> {out}  ({LENGTHS[0]} x {LENGTHS[1]} x {LENGTHS[2]} bar)", flush=True)
