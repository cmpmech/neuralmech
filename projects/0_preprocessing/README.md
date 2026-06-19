# Preprocessing

One-off data-generation utilities that turn raw external datasets (in
`external_data/`) into the arrays and meshes for chapter drivers. Each
script is self-contained. Its output is what the rest of the
repo reads. Nothing here feeds a book figure directly.

- `bunny_pointcloud.py` -> `data/bunny_pointcloud.npz`, `results/3D/bunny_pointcloud.vtk`
  voxel-downsamples the Stanford bunny PLY to one point per occupied voxel; writes
  a torch `.npz` and a ParaView point cloud
- `extract_ct_2D.py` -> `data/<name>_2D.npz`
  binarizes, crops, and largest-component-cleans the central slice of a CT volume
  into a 2D indicator field
- `extract_ct_3D.py` -> `data/<name>_3D.npz`
  the 3D counterpart: binarizes, crops, and cleans the full CT volume into a
  voxel indicator field
- `extract_rock_slices.py` -> `data/rocks_<name>_320.npy`, `results/3D/<name>.vtr`
  crops a fixed window from each rock CT scan and samples slices through it; also
  exports the volume for ParaView
- `mesh_ghana.py` -> `data/ghana_mesh.pt`
  extracts the Ghana outline from the Natural Earth shapefile and triangulates it
  into a graph (node positions, edges, triangles) for the graph-network examples
