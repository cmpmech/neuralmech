from pathlib import Path

from paraview.simple import *
from PIL import Image

BASE_DIR = Path(__file__).parent
STL_DIR = BASE_DIR / "../../data/abc/geometry/stl"
VOXEL_DIR = BASE_DIR / "../../results/abc/geometry"
RENDER_DIR = BASE_DIR / "../../results/abc/renders"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

# square aspect avoids cropping wide parts; ZOOM < 1 leaves a margin around the part
RESOLUTION = [1000, 1000]
ZOOM = 1.0
# render at SUPERSAMPLE x RESOLUTION then downscale, to anti-alias the high-frequency
# voxel step facets (FXAA alone only smooths silhouettes, not surface sparkle)
SUPERSAMPLE = 3

# run with pvpython


def render_and_save(
    source,
    out_name,
    metallic=0.3,
    key_intensity=0.8,
    fill_ratio=3.0,
    key_elevation=50,  # lower = more grazing light, reveals surface relief/voxel steps
    interpolation="PBR",  # "Flat"/"Gouraud" (specular only applies to these two)
    specular=0.0,  # glint strength; catches step faces at grazing angles
    specular_power=20,  # highlight tightness, higher = smaller/sharper glint
    color=(0.40, 0.50, 0.82),  # blueish; (0.80, 0.30, 0.25) is reddish
):
    # Display
    display = Show(source)
    display.SetRepresentationType("Surface")
    # the voxel reader auto-colors by its "indicator" cell scalar, which overrides
    # DiffuseColor; turn that off (only when active) so the solid color applies
    if display.ColorArrayName.GetArrayName():
        ColorBy(display, None)

    # PBR shading gives the surface a material feel instead of flat gray:
    # roughness controls matte (1.0) vs glossy (0.0); metallic ~0 is plastic/clay
    display.Interpolation = interpolation
    display.Metallic = metallic
    display.Roughness = 0.4  # 0.45
    display.DiffuseColor = list(color)

    # specular highlight (Flat/Gouraud only; PBR uses roughness/metallic instead)
    display.Specular = specular
    display.SpecularPower = specular_power

    renderView1 = GetActiveViewOrCreate("RenderView")

    # Three-point light kit: key light pushed off-camera reveals curvature;
    # fill softens shadows, back light rims the part. Azimuth is the main knob.
    renderView1.UseLight = 1
    renderView1.KeyLightIntensity = key_intensity
    renderView1.KeyLightElevation = key_elevation  # degrees above horizon
    renderView1.KeyLightAzimuth = -35  # degrees off-camera (sideways)
    renderView1.KeyLightWarmth = 0.55  # 0.5 neutral, > 0.5 warmer
    renderView1.FillLightKFRatio = fill_ratio  # key:fill ratio, lower = brighter fill
    renderView1.BackLightKBRatio = 3.5
    renderView1.HeadLightKHRatio = 3.0  # keep headlight weak so it is not flat
    renderView1.MaintainLuminance = 1

    # Tone mapping balances exposure and avoids blown-out highlights
    renderView1.UseToneMapping = 1

    # Remove orientation axes (coordinate system widget)
    renderView1.OrientationAxesVisibility = 0

    # Transparent background
    renderView1.Background = [1, 1, 1]  # white, or ignored when transparent
    renderView1.UseColorPaletteForBackground = 0

    renderView1.ResetCamera()
    camera = GetActiveCamera()
    camera.SetPosition(1, -0.5, 0.5)  # view from +X
    camera.SetViewUp(0, 0, 1)  # Z points up
    camera.SetFocalPoint(0, 0, 0)
    renderView1.ResetCamera()
    camera.Zoom(ZOOM)
    Render()

    # Save with transparent background, supersampled, then downscale to anti-alias
    out_path = str(RENDER_DIR / f"{out_name}.png")
    SaveScreenshot(
        out_path,
        renderView1,
        ImageResolution=[SUPERSAMPLE * r for r in RESOLUTION],
        TransparentBackground=1,
    )
    Image.open(out_path).resize(RESOLUTION, Image.LANCZOS).save(out_path)


for STL_IDX in [3, 4, 5, 13, 14, 15]:
    STL_NAME = f"{STL_IDX:09}_abc"

# ----------------------------------- STL ------------------------------------
    reader = STLReader(FileNames=[str(STL_DIR / f"{STL_NAME}.stl")])
    reader.UpdatePipeline()
    render_and_save(reader, STL_NAME)
    Delete(reader)
    del reader

# -------------------------------- voxel vti ---------------------------------
    voxel = XMLImageDataReader(FileName=[str(VOXEL_DIR / f"{STL_NAME}.vti")])
    voxel.UpdatePipeline()

    # keep only the inside voxels (indicator in [1, 255]); 0 is background
    threshold = Threshold(Input=voxel)
    threshold.Scalars = ["CELLS", "indicator"]
    threshold.LowerThreshold = 1
    threshold.UpperThreshold = 255
    threshold.UpdatePipeline()

    # flat axis-aligned voxel faces catch little of the off-axis key light, so
    # render them lighter and non-metallic; keep some fill contrast so the cubic
    # faces still read as 3D rather than flat
    render_and_save(
        threshold,
        f"{STL_NAME}_voxel",
        metallic=0.1,
        key_intensity=1.0,
        fill_ratio=4,
        interpolation="PBR",  # hard per-face normals so the voxel steps read crisply
        color=(0.4, 0.5, 0.8),  # lighter blue
    )
    # render_and_save(
    #     threshold,
    #     f"{STL_NAME}_voxel",
    #     metallic=0.0,
    #     key_intensity=1.0,
    #     fill_ratio=3,
    #     interpolation="Gouraud",  # smooth shading + specular glint to define relief
    #     specular=0.8,
    #     specular_power=80,
    #     color=(0.6, 0.6, 0.6),  # lighter blue
    # )

    Delete(threshold)
    Delete(voxel)
    del threshold, voxel
