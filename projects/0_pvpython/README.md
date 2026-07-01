# ParaView Rendering

Render scripts that turn 3D `results/` data (STL, VTI, VTK, VTR) into transparent PNGs
for the book and slides. They run under ParaView's `pvpython`:

```bash
pvpython render_abc.py
```

Each writes its renders into `results/rgb_png/` under their book-final names (a committed
folder, assumed to exist), so `projects/0_cmyk_export/figures_to_cmyk.py` derives the
matching CMYK print pdfs in `results/cmyk_pdf/`.

- `render_abc.py` -> `rgb_png/<id>_abc.png`, `<id>_abc_voxel.png`
  PBR surface renders of the ABC geometries and their voxelizations, lit with a
  three-point light kit
- `render_pointnet_bunny.py` -> `rgb_png/pointnet_bunny_<field>.png`
  the PointNet bunny as a sphere-glyph point cloud, one render per scalar field
  (coordinates, ground truth, prediction)
- `render_rocks.py` -> `rgb_png/rocks_<NAME>.png`
  a rock volume rendered with the X-ray transfer function
- `pvpython_helper.py`
  shared helpers: view cleanup, camera framing/orbit, `.cmap` preset registration,
  and supersampled transparent-PNG saving
