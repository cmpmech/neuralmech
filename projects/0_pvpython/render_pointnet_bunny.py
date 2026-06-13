import json
import math
from pathlib import Path

from paraview.servermanager import vtkSMTransferFunctionPresets
from paraview.simple import *
from PIL import Image

BASE_DIR = Path(__file__).parent
VTK_FILE = BASE_DIR / "../../results/3D/pointnet_bunny.vtk"
RENDER_DIR = BASE_DIR / "../../results/3D/renders"
CMAP_DIR = BASE_DIR / "../../.cmap"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

# run with pvpython

# ----------------------------------- settings ----------------------------------------
POINT_SIZE = 15.0  # point diameter in final-image pixels
SPHERE_RESOLUTION = 16  # theta/phi tessellation of each sphere glyph
ROTATION_Y = -20.0  # view rotation about the bunny's vertical (Y) axis, degrees
RESOLUTION = [1200, 1200]
# render at SUPERSAMPLE x RESOLUTION then downscale to anti-alias the points
SUPERSAMPLE = 3

# pick one for COLORMAP. viridis/inferno/plasma/turbo/jet/coolwarm are native ParaView
# presets; Spectral and rainbow are loaded from the .cmap/ folder (matplotlib's Spectral
# only ships as a categorical preset in ParaView, so we carry it as RGBPoints instead)
COLORMAP = "Spectral"
COLORMAPS = {
    "Spectral": "Spectral",  # .cmap/spectral.cmap
    "rainbow": "Rainbow Desaturated",  # .cmap/rainbow_desaturated.cmap
    "viridis": "Viridis (matplotlib)",
    "inferno": "Inferno (matplotlib)",
    "plasma": "Plasma (matplotlib)",
    "coolwarm": "Cool to Warm (Extended)",
    "turbo": "Turbo",
    "jet": "Jet",
}

# (ColorBy argument, output name); x/y/z come from the "input" coordinate vector
FIELDS = [
    (("POINTS", "input", "X"), "x"),
    (("POINTS", "input", "Y"), "y"),
    (("POINTS", "input", "Z"), "z"),
    (("POINTS", "ground_truth"), "ground_truth"),
    (("POINTS", "prediction"), "prediction"),
]

# ------------------------------------- load ------------------------------------------
reader = LegacyVTKReader(FileNames=[str(VTK_FILE)])
reader.UpdatePipeline()

# register every .cmap (ParaView JSON) in the folder as a named preset
presets = vtkSMTransferFunctionPresets.GetInstance()
for cmap_file in sorted(CMAP_DIR.glob("*.cmap")):
    text = cmap_file.read_text()
    presets.AddPreset(json.loads(text)["Name"], text)

# ------------------------------------ display ----------------------------------------
# real sphere geometry (one tessellated sphere per point) rather than point sprites:
# impostor spheres (RenderPointsAsSpheres) depth-fight where they overlap, leaving
# transparent speckles; actual geometry intersects correctly
bounds = reader.GetDataInformation().GetBounds()
diag = math.dist(bounds[0::2], bounds[1::2])

glyph = Glyph(Input=reader, GlyphType="Sphere")
glyph.GlyphMode = "All Points"
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.GlyphType.ThetaResolution = SPHERE_RESOLUTION
glyph.GlyphType.PhiResolution = SPHERE_RESOLUTION
glyph.ScaleFactor = 0.003 * diag  # small placeholder so ResetCamera frames the points

renderView1 = GetActiveViewOrCreate("RenderView")
display = Show(glyph, renderView1)
display.SetRepresentationType("Surface")
display.Specular = 0.0  # no glossy white glint

renderView1.OrientationAxesVisibility = 0
renderView1.Background = [1, 1, 1]
renderView1.UseColorPaletteForBackground = 0

# bunny height runs along +Y; view from +Z (front)
renderView1.ResetCamera()
camera = GetActiveCamera()
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 1, 0)
camera.SetPosition(0, 0, 1)
camera.Azimuth(ROTATION_Y)  # rotate about view-up (Y) axis
renderView1.ResetCamera()

# size spheres so POINT_SIZE is the diameter in final-image px (exact at the focal
# plane; perspective makes nearer spheres slightly larger). ScaleFactor = world diameter
dist = math.dist(camera.GetPosition(), camera.GetFocalPoint())
world_per_px = (
    2 * dist * math.tan(math.radians(camera.GetViewAngle()) / 2) / RESOLUTION[1]
)
glyph.ScaleFactor = POINT_SIZE * world_per_px

# ------------------------------------ render -----------------------------------------
for color_by, name in FIELDS:
    ColorBy(display, color_by)
    lut = GetColorTransferFunction(color_by[1])
    lut.ApplyPreset(COLORMAPS[COLORMAP], True)
    display.RescaleTransferFunctionToDataRange(False, True)  # force exact field range
    HideScalarBarIfNotNeeded(lut, renderView1)
    Render()

    out_path = str(RENDER_DIR / f"pointnet_bunny_{name}.png")
    SaveScreenshot(
        out_path,
        renderView1,
        ImageResolution=[SUPERSAMPLE * r for r in RESOLUTION],
        TransparentBackground=1,
    )
    Image.open(out_path).resize(RESOLUTION, Image.LANCZOS).save(out_path)
    print(f"\tsaved {out_path}")
