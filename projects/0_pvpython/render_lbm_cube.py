# run with pvpython
from pathlib import Path

import numpy as np
from paraview.simple import *
from PIL import Image
from pvpython_helper import clean_view, register_cmaps, save_png
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RGB_PNG_DIR = (BASE_DIR / "../../results/rgb_png").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()

# ----------------------------------- settings ----------------------------------------
Q_LEVEL = 30  # Q-criterion over (inflow velocity / cube edge)^2
WALL_LAYERS = 6  # nodes next to the cube without isosurfaces, staircase artifacts
BODY_COLOR = [0.92, 0.90, 0.84]  # ivory
AZIMUTH, ELEVATION = 35, 25  # camera upstream of the cube, from the side and above
RESOLUTION = [3200, 1800]

# --------------------------------------- render --------------------------------------
data = np.load(DATA_DIR / "lbm_cube.npz")
solid = data["solid"]
q = data["q"].astype(np.float32)
near = solid.copy()
for _ in range(WALL_LAYERS):
    for axis in range(3):
        near |= np.roll(near, 1, axis) | np.roll(near, -1, axis)
q[near] = 0
q[:3] = q[-3:] = 0  # inlet and outlet planes
vorticity_x = data["vorticity_x"].astype(np.float32)
scale = 1.5 * float(np.percentile(abs(vorticity_x[~solid]), 99.5))

grid = vtkImageData()
grid.SetDimensions(*solid.shape)
for name, array in (("q", q), ("vorticity_x", vorticity_x), ("solid", solid)):
    values = numpy_to_vtk(np.asarray(array, dtype=np.float32).ravel(order="F"), deep=True)
    values.SetName(name)
    grid.GetPointData().AddArray(values)
source = PVTrivialProducer()
source.GetClientSideObject().SetOutput(grid)

view = GetActiveViewOrCreate("RenderView")
clean_view(view)
register_cmaps(CMAP_DIR)

body = Show(Contour(Input=source, ContourBy=["POINTS", "solid"], Isosurfaces=[0.5]), view)
ColorBy(body, None)
body.DiffuseColor = body.AmbientColor = BODY_COLOR
body.Specular = 0.0

vortices = Contour(Input=source, ContourBy=["POINTS", "q"], Isosurfaces=[Q_LEVEL])
vortices.ComputeScalars = 0
display = Show(vortices, view)
ColorBy(display, ("POINTS", "vorticity_x"))
display.Specular = 0.0
lut = GetColorTransferFunction("vorticity_x")
lut.ApplyPreset("guppy", True)
lut.AutomaticRescaleRangeMode = "Never"
lut.RescaleTransferFunction(-scale, scale)
HideScalarBarIfNotNeeded(lut, view)

center = [n / 2 for n in solid.shape]
camera = GetActiveCamera()
camera.SetFocalPoint(*center)
camera.SetViewUp(0, 0, 1)
camera.SetPosition(center[0], center[1] + 3 * solid.shape[0], center[2])
camera.Azimuth(AZIMUTH)
camera.Elevation(ELEVATION)
view.ResetCamera()
Render()

save_png(RGB_PNG_DIR / "lbm_cube.png", view, RESOLUTION, supersample=2)
render = Image.open(RGB_PNG_DIR / "lbm_cube.png")
render.crop(render.getchannel("A").getbbox()).save(RGB_PNG_DIR / "lbm_cube.png")  # drop empty margins
