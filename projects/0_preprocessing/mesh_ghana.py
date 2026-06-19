from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
import triangle as tr
from rasterio.features import rasterize
from rasterio.transform import from_bounds

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

# ----------------------------------- load map data -----------------------------------
gdf = gpd.read_file(DATA_DIR / "ne_50m_admin_0_countries.shp")

# ----------------------------------- extract ghana -----------------------------------
ghana = gdf[gdf["ISO_A3"] == "GHA"]
print(ghana[["ADMIN", "ISO_A3", "CONTINENT"]])
ghana = ghana.dissolve()
ghana = ghana.to_crs("EPSG:3857")

fig = ghana.plot(facecolor="black")
fig.set_axis_off()
plt.show()

# ---------------------------------- rasterize ghana ----------------------------------
# dx = dy = 5000  # m
# minx, miny, maxx, maxy = ghana.total_bounds

# width = int((maxx - minx) / dx)
# height = int((maxy - miny) / dy)

# transform = from_bounds(minx, miny, maxx, maxy, width, height)

# grid = rasterize(
#     [(geom, 1) for geom in ghana.geometry],
#     out_shape=(height, width),
#     transform=transform,
#     fill=0,
#     dtype=np.uint8,
# )

# fig, ax = plt.subplots()
# ax.imshow(grid, cmap="Greys")
# ax.set_axis_off()
# plt.show()

# --------------------------------- triangulate ghana ---------------------------------
geom = ghana.geometry.iloc[0]
coords = np.array(geom.exterior.coords[:-1])
coords[:, 0] -= np.min(coords[:, 0])
coords[:, 1] -= np.min(coords[:, 1])
coords /= np.max(np.abs(coords))

segments = np.array([(i, (i + 1) % len(coords)) for i in range(len(coords))])
tri_input = {"vertices": coords, "segments": segments}

# p: use segments, q25: minimum angle 25 deg, a: max area
tri_output = tr.triangulate(tri_input, "pq25a0.0004")  # fine mesh

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.triplot(
    tri_output["vertices"][:, 0],
    tri_output["vertices"][:, 1],
    tri_output["triangles"],
    "b-",
    linewidth=0.5,
)
ax.plot(tri_output["vertices"][:, 0], tri_output["vertices"][:, 1], "ro", markersize=2)
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()

# --------------------------------------- export --------------------------------------
triangles = tri_output["triangles"]
edges = []
for tri in triangles:
    edges.extend([[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]])
    edges.extend(
        [[tri[1], tri[0]], [tri[2], tri[1]], [tri[0], tri[2]]]
    )  # bidirectional

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
pos = torch.tensor(tri_output["vertices"], dtype=torch.float)

torch.save(
    {
        "edge_index": edge_index,  # [2, edges]
        "pos": pos,  # [nodes, 2]
        "triangles": torch.tensor(triangles, dtype=torch.long),
    },
    DATA_DIR / "ghana_mesh.pt",
)
