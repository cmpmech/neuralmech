from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# -------------------------------------- settings -------------------------------------
# geometry
LENGTH = 1.0
HOLE_CENTER = [0.5, 0.5]
HOLE_RADIUS = 0.15

# discretization
ELEMENT_SIZE = 0.03

# postprocessing
MESH_COLOR = to_rgba("gray", 0.8)  # darker than the solvers, nothing behind the lines
MESH_WIDTH = 0.2

# ---------------------------------------- mesh ---------------------------------------
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("plate_with_hole")

plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, LENGTH, LENGTH)
hole = gmsh.model.occ.addDisk(
    HOLE_CENTER[0], HOLE_CENTER[1], 0.0, HOLE_RADIUS, HOLE_RADIUS
)
gmsh.model.occ.cut([(2, plate)], [(2, hole)])
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", ELEMENT_SIZE)
gmsh.option.setNumber("Mesh.MeshSizeMax", ELEMENT_SIZE)
gmsh.option.setNumber("Mesh.Algorithm", 8)  # frontal-delaunay for quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.generate(2)

tags, flat_coords, _ = gmsh.model.mesh.getNodes()
_, flat_quads = gmsh.model.mesh.getElementsByType(3)  # 3 is the 4-node quadrilateral
gmsh.finalize()

order = np.argsort(tags)
coords = flat_coords.reshape(-1, 3)[order, :2]
renumber = np.zeros(tags.max() + 1, dtype=int)
renumber[tags[order]] = np.arange(len(tags))
quads = renumber[flat_quads.reshape(-1, 4)]

# --------------------------------------- export --------------------------------------
np.savez(
    DATA_DIR / "plate_with_hole.npz",
    coords=coords,
    quads=quads,
    length=LENGTH,
    hole_center=HOLE_CENTER,
    hole_radius=HOLE_RADIUS,
)
print(f"{len(coords)} nodes, {len(quads)} quadrilaterals")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.add_collection(
    PolyCollection(
        coords[quads], facecolors="none", edgecolors=MESH_COLOR, linewidths=MESH_WIDTH
    )
)

ax.autoscale()
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
