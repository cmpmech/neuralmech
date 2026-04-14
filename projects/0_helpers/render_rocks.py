from pathlib import Path

from paraview.simple import *

BASE_DIR = Path(__file__).parent


# name = "B-HAI-1"
# name = "KAK-1"
# name = "KAK-2"
# name = "BM-5"
# name = "BM-48"
# name = "BM-105"
# name = "WD-2"
name = "WD-150"


# run with pvpython

# Load
reader = XMLRectilinearGridReader(
    FileName=[str(BASE_DIR / f"../../results/3D/{name}.vtr")]
)
reader.UpdatePipeline()

# Display
display = Show(reader)
display.SetRepresentationType("Surface")
ColorBy(display, ("CELLS", "scalar"))

renderView1 = GetActiveViewOrCreate("RenderView")
scalarLUT = GetColorTransferFunction("scalar")
scalarLUT.ApplyPreset("X Ray", True)
renderView1.ApplyIsometricView()

# Remove colorbar
HideScalarBarIfNotNeeded(scalarLUT, renderView1)

# Remove orientation axes (coordinate system widget)
renderView1.OrientationAxesVisibility = 0

# Transparent background
renderView1.Background = [1, 1, 1]  # white, or ignored when transparent
renderView1.UseColorPaletteForBackground = 0

# GetActiveView().ResetCamera()

# renderView1.ResetCamera()
# camera = GetActiveCamera()
# camera.Elevation(90)
# Render()

renderView1.ResetCamera()
camera = GetActiveCamera()
# camera.SetPosition(1, 0, 0)  # front view?
camera.SetPosition(1, -0.5, 0.5)  # view from +X
camera.SetViewUp(0, 0, 1)  # Z points up
camera.SetFocalPoint(0, 0, 0)
renderView1.ResetCamera()
Render()

# Save with transparent background
SaveScreenshot(
    str(BASE_DIR / f"../../results/3D/renders/{name}.png"),
    renderView1,
    ImageResolution=[500, 1000],
    TransparentBackground=1,
)


# for i, angle in enumerate(np.linspace(0, 360, 120)):
#     rad = np.radians(angle)
#     camera.SetPosition(np.cos(rad), np.sin(rad), 0.3)
#     camera.SetViewUp(0, 0, 1)
#     camera.SetFocalPoint(0, 0, 0)
#     renderView1.ResetCamera()
#     Render()
#     SaveScreenshot(f"frames/frame_{i:04d}.png", renderView1, ImageResolution=[1920, 1080], TransparentBackground=1)
