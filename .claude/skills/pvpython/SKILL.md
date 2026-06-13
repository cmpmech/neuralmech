---
name: pvpython
description: Writing or editing pvpython/ParaView rendering scripts in `code/projects/0_pvpython/`. Covers the render pipeline (reader → filter → Show → camera → colormap → screenshot), point-cloud rendering with real sphere glyphs (avoiding impostor speckles), colormap handling including the shared `.cmap/` folder and the Spectral pitfall, supersampled anti-aliasing, transparent backgrounds, camera/rotation, and legacy-VTK point-cloud export. Invoke before writing or editing any pvpython script.
---

pvpython scripts render `results/` data (VTK/VTU/VTR/STL) to PNGs for the book and
slides. They run under **ParaView's `pvpython`**, not the project venv — a separate
interpreter that bundles its own numpy/VTK. Run with `pvpython script.py`.

## Where scripts live & house style

`code/projects/0_pvpython/<render_name>.py`. Follow the repo driver style: bare
scaffolding (imports → `BASE_DIR`/paths), section headers only from settings
onward, no `__main__` guard, no aligned `=`. Mark the interpreter with a top
comment `# run with pvpython`. Reference scripts:

- `render_pointnet_bunny.py` — point cloud → sphere glyphs, `.cmap` colormaps.
- `render_abc.py` — PBR surfaces, three-point lighting, supersampled output.
- `render_rocks.py` — volumetric `.vtr`, single preset, isometric view.

## Imports & paths

```python
import json
import math
from pathlib import Path

from paraview.servermanager import vtkSMTransferFunctionPresets
from paraview.simple import *
from PIL import Image

BASE_DIR = Path(__file__).parent
```

`from paraview.simple import *` is the expected idiom here (don't fight it).
`PIL` is available in pvpython and used for the downscale step.

## Canonical render pipeline

```python
reader = LegacyVTKReader(FileNames=[str(VTK_FILE)])   # or XML*Reader / STLReader
reader.UpdatePipeline()

renderView1 = GetActiveViewOrCreate("RenderView")
display = Show(source, renderView1)
display.SetRepresentationType("Surface")    # or "Points"
display.Specular = 0.0                       # kill glossy white glints

renderView1.OrientationAxesVisibility = 0
renderView1.Background = [1, 1, 1]
renderView1.UseColorPaletteForBackground = 0

renderView1.ResetCamera()
camera = GetActiveCamera()
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 1, 0)        # match the data's up-axis
camera.SetPosition(0, 0, 1)
camera.Azimuth(ROTATION_Y)       # rotate about the view-up axis (degrees)
renderView1.ResetCamera()

ColorBy(display, ("POINTS", "field"))
lut = GetColorTransferFunction("field")
lut.ApplyPreset(PRESET_NAME, True)
display.RescaleTransferFunctionToDataRange(False, True)   # extend=False, force=True
HideScalarBarIfNotNeeded(lut, renderView1)
Render()
```

- Color a vector component with a 3-tuple: `ColorBy(display, ("POINTS", "input", "X"))`;
  the LUT name is still the array name (`"input"`).
- `RescaleTransferFunctionToDataRange(False, True)` is the reliable form — `extend=True`
  only grows the range (leaves a stale range → everything maps to one color).

## Point clouds: render real spheres, not impostors

`display.RenderPointsAsSpheres = 1` (on the `Points` representation) draws **flat
billboard impostors** that fake depth in a shader. Where two impostors overlap their
faked depths fight and fragments get discarded → transparent/white speckles at every
sphere intersection (isolated spheres look fine — that's the tell). No amount of
`Specular=0`, supersampling, tone mapping, or projection change fixes it; it is a
depth bug, not an edge bug. OSPRay ray tracing would fix it but is often not built
into the installed ParaView (`Refusing to enable OSPRay...`).

Render **actual sphere geometry** with a `Glyph` instead:

```python
glyph = Glyph(Input=reader, GlyphType="Sphere")
glyph.GlyphMode = "All Points"               # default subsamples to 5000 points!
glyph.ScaleArray = ["POINTS", "No scale array"]   # uniform size, don't scale by data
glyph.GlyphType.ThetaResolution = SPHERE_RESOLUTION
glyph.GlyphType.PhiResolution = SPHERE_RESOLUTION
# size in world units; convert a pixel diameter via the focal-plane scale:
dist = math.dist(camera.GetPosition(), camera.GetFocalPoint())
world_per_px = 2 * dist * math.tan(math.radians(camera.GetViewAngle()) / 2) / RESOLUTION[1]
glyph.ScaleFactor = POINT_SIZE * world_per_px     # ScaleFactor = sphere diameter
```

Set a small placeholder `ScaleFactor` (e.g. `0.003 * bbox_diag`) **before** the
camera `ResetCamera` so the points (not giant default glyphs) frame the view, then
set the real `ScaleFactor` after. The `Glyph` copies point data through, so `ColorBy`
works on the glyph display unchanged.

## Colormaps

Native matplotlib presets ship under parenthesized names:
`"Viridis (matplotlib)"`, `"Inferno (matplotlib)"`, `"Plasma (matplotlib)"`,
`"Magma (matplotlib)"`. Also `"Cool to Warm (Extended)"`, `"Turbo"`, `"Jet"`,
`"Rainbow Desaturated"`.

**Spectral pitfall:** ParaView only ships Spectral as a *categorical* preset
(`"Brewer Diverging Spectral (11)"`) whose colors live in `IndexedColors`, not
`RGBPoints`. `ApplyPreset` leaves the continuous `RGBPoints` at the default
blue→gray→red, so continuous data renders flat/wrong. Carry a continuous Spectral as
an `RGBPoints` `.cmap` instead.

### The shared `.cmap/` folder

`code/.cmap/*.cmap` are ParaView-JSON colormap files (`{"Name", "ColorSpace",
"RGBPoints":[x,r,g,b, ...]}`) shared between pvpython and matplotlib. Register them
as presets by **content**, not via `ImportPresets` (which dispatches on a `.json`/
`.xml` extension and rejects `.cmap`):

```python
presets = vtkSMTransferFunctionPresets.GetInstance()
for cmap_file in sorted(CMAP_DIR.glob("*.cmap")):
    text = cmap_file.read_text()
    presets.AddPreset(json.loads(text)["Name"], text)   # then ApplyPreset(name, True)
```

The same files load in matplotlib via `postprocessing.load_cmap` (see the
`create_project_driver` skill). When adding a new shared colormap, dump a ParaView
preset with `vtkSMTransferFunctionPresets.GetPresetAsString(i)` and keep
`{Name, ColorSpace, RGBPoints}`.

## Anti-aliasing & saving

Render at `SUPERSAMPLE × RESOLUTION` then downscale with PIL — FXAA/MSAA only smooth
silhouettes, not high-frequency surface/point sparkle:

```python
SaveScreenshot(out_path, renderView1,
               ImageResolution=[SUPERSAMPLE * r for r in RESOLUTION],
               TransparentBackground=1)
Image.open(out_path).resize(RESOLUTION, Image.LANCZOS).save(out_path)
```

`Background = [1, 1, 1]` + `TransparentBackground=1` gives a transparent PNG (it may
display black in some image viewers, but corner alpha is 0). PIL's LANCZOS resize is
non-premultiplied, so a *white* transparent background can bleed faint white into
edges; if that shows, premultiply alpha before resizing (or render on the final
background color). For `Points`/`PointSize` (pixels), multiply by `SUPERSAMPLE`; for
glyph world-unit sizes nothing extra is needed.

## Writing point clouds for ParaView (from the project venv side)

Legacy ASCII VTK `POLYDATA` with a `VERTICES` block renders points directly (no
filter needed); add a `POINT_DATA` block for fields. See
`projects/0_preprocessing/bunny_pointcloud.py` (geometry only) and
`projects/4_specialized_nns/pointnet_example.py` (multiple scalar fields + a vector):

```
# vtk DataFile Version 3.0
<title>
ASCII
DATASET POLYDATA
POINTS <M> float
<x y z> ...
VERTICES <M> <2*M>
1 <i> ...
POINT_DATA <M>
SCALARS <name> float 1
LOOKUP_TABLE default
<v> ...
VECTORS <name> float
<x y z> ...
```
