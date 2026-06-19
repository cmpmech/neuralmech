# run with pvpython
"""Shared helpers for the 0_pvpython render scripts."""
import json

from paraview.simple import GetActiveCamera, SaveScreenshot
from paraview.servermanager import vtkSMTransferFunctionPresets
from PIL import Image


def clean_view(view):
    """Hide the orientation widget; white background (transparent on save)."""
    view.OrientationAxesVisibility = 0
    view.Background = [1, 1, 1]
    view.UseColorPaletteForBackground = 0


def view_from_x(view):
    """Point the camera from +X (Z up), framed on the data; return the camera."""
    view.ResetCamera()
    camera = GetActiveCamera()
    camera.SetPosition(1, -0.5, 0.5)
    camera.SetViewUp(0, 0, 1)
    camera.SetFocalPoint(0, 0, 0)
    view.ResetCamera()
    return camera


def orbit(view, azimuth=0.0, elevation=0.0, roll=0.0):
    """Rotate the camera about the focal point and refit; return the camera.

    From a front view the angles (deg) map to world axes: azimuth=view-up,
    elevation=horizontal screen axis, roll=view axis (Y/X/Z when view-up is +Y).
    """
    camera = GetActiveCamera()
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    camera.Roll(roll)
    view.ResetCamera()
    return camera


def register_cmaps(cmap_dir):
    """Register every .cmap in cmap_dir as a named preset (by content, since
    ImportPresets rejects the extension). Apply with lut.ApplyPreset(name, True)."""
    presets = vtkSMTransferFunctionPresets.GetInstance()
    for cmap_file in sorted(cmap_dir.glob("*.cmap")):
        text = cmap_file.read_text()
        presets.AddPreset(json.loads(text)["Name"], text)


def save_png(out_path, view, resolution, supersample=1):
    """Save view as a transparent PNG; render at supersample x resolution and
    downscale (LANCZOS) to anti-alias. supersample=1 skips the resize."""
    out_path = str(out_path)
    SaveScreenshot(out_path, view,
                   ImageResolution=[supersample * r for r in resolution],
                   TransparentBackground=1)
    if supersample != 1:
        Image.open(out_path).resize(resolution, Image.LANCZOS).save(out_path)
