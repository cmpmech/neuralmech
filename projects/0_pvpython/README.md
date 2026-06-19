# ParaView Rendering

Render scripts that turn 3D `results/` data (STL, VTI, VTK, VTR) into PNGs for the
book and slides. They run under ParaView's `pvpython`:

```bash
pvpython render_abc.py
```

- `render_abc.py` -> `results/abc/renders/`
  PBR surface renders of the ABC geometries and their voxelizations, lit with a
  three-point light kit
- `render_pointnet_bunny.py` -> `results/3D/renders/`
  the PointNet bunny as a sphere-glyph point cloud, one render per scalar field
  (coordinates, ground truth, prediction)
- `render_rocks.py` -> `results/3D/renders/`
  a rock volume rendered with the X-ray transfer function
- `pvpython_helper.py`
  shared helpers: view cleanup, camera framing/orbit, `.cmap` preset registration,
  and supersampled transparent-PNG saving
