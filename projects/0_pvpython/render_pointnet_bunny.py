# run with pvpython
import math
from pathlib import Path

from paraview.simple import *
from pvpython_helper import clean_view, orbit, register_cmaps, save_png

BASE_DIR = Path(__file__).parent
VTK_PATH = (BASE_DIR / "../../results/3D/pointnet_bunny.vtk").resolve()
RENDER_DIR = (BASE_DIR / "../../results/rgb_png").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()

# ----------------------------------- settings ----------------------------------------
POINT_SIZE = 15.0  # sphere diameter in final-image pixels
SPHERE_RESOLUTION = 16
ROTATION_Y = -20.0  # camera azimuth about the bunny's vertical (Y) axis, degrees
RESOLUTION = [1200, 1200]
SUPERSAMPLE = 3
COLORMAP = "Spectral"  # carried in .cmap/ (ParaView only ships Spectral as categorical)

# (ColorBy argument, output name)
FIELDS = [
    (("POINTS", "input", "X"), "x"),
    (("POINTS", "input", "Y"), "y"),
    (("POINTS", "input", "Z"), "z"),
    (("POINTS", "ground_truth"), "ground_truth"),
    (("POINTS", "prediction"), "prediction"),
]

# ------------------------------------- load ------------------------------------------
reader = LegacyVTKReader(FileNames=[str(VTK_PATH)])
reader.UpdatePipeline()
register_cmaps(CMAP_DIR)

# ------------------------------------ display ----------------------------------------
# real sphere geometry per point, not point sprites: impostors depth-fight where they
# overlap, leaving transparent speckles
bounds = reader.GetDataInformation().GetBounds()
diag = math.dist(bounds[0::2], bounds[1::2])

glyph = Glyph(Input=reader, GlyphType="Sphere")
glyph.GlyphMode = "All Points"
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.GlyphType.ThetaResolution = SPHERE_RESOLUTION
glyph.GlyphType.PhiResolution = SPHERE_RESOLUTION
glyph.ScaleFactor = 0.003 * diag  # placeholder so ResetCamera frames the points

renderView1 = GetActiveViewOrCreate("RenderView")
display = Show(glyph, renderView1)
display.SetRepresentationType("Surface")
display.Specular = 0.0
clean_view(renderView1)

# view from +Z (front, Y up), then azimuth about Y
renderView1.ResetCamera()
camera = GetActiveCamera()
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 1, 0)
camera.SetPosition(0, 0, 1)
camera = orbit(renderView1, azimuth=ROTATION_Y)

# size spheres so POINT_SIZE is their diameter in px at the focal plane
dist = math.dist(camera.GetPosition(), camera.GetFocalPoint())
world_per_px = (
    2 * dist * math.tan(math.radians(camera.GetViewAngle()) / 2) / RESOLUTION[1]
)
glyph.ScaleFactor = POINT_SIZE * world_per_px

# ------------------------------------ render -----------------------------------------
for color_by, name in FIELDS:
    ColorBy(display, color_by)
    lut = GetColorTransferFunction(color_by[1])
    lut.ApplyPreset(COLORMAP, True)
    display.RescaleTransferFunctionToDataRange(False, True)
    HideScalarBarIfNotNeeded(lut, renderView1)
    Render()

    save_png(
        RENDER_DIR / f"pointnet_bunny_{name}.png", renderView1, RESOLUTION, SUPERSAMPLE
    )
    print(f"\tsaved pointnet_bunny_{name}.png")
