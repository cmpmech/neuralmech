# run with pvpython
from pathlib import Path

from paraview.simple import *
from pvpython_helper import clean_view, save_png, view_from_x

BASE_DIR = Path(__file__).parent
STL_DIR = (BASE_DIR / "../../data/abc/geometry/stl").resolve()
VOXEL_DIR = (BASE_DIR / "../../results/abc/geometry").resolve()
RENDER_DIR = (BASE_DIR / "../../results/rgb_png").resolve()

# ----------------------------------- settings ----------------------------------------
RESOLUTION = [1000, 1000]
SUPERSAMPLE = 3  # render at SUPERSAMPLE x RESOLUTION then downscale, to anti-alias
STL_INDICES = [3, 4, 5, 13, 14, 15]


# ------------------------------------- helper ----------------------------------------
def render_and_save(source, out_name, metallic=0.3, key_intensity=0.8, fill_ratio=3.0,
                    color=(0.40, 0.50, 0.82)):
    display = Show(source)
    display.SetRepresentationType("Surface")
    # the voxel reader auto-colors by its "indicator" scalar, overriding DiffuseColor
    if display.ColorArrayName.GetArrayName():
        ColorBy(display, None)

    # PBR: roughness 1=matte / 0=glossy, metallic ~0=plastic
    display.Interpolation = "PBR"
    display.Metallic = metallic
    display.Roughness = 0.4
    display.DiffuseColor = list(color)

    renderView1 = GetActiveViewOrCreate("RenderView")
    # three-point light kit: key reveals curvature, fill softens shadows, back rims it
    renderView1.UseLight = 1
    renderView1.KeyLightIntensity = key_intensity
    renderView1.KeyLightElevation = 50
    renderView1.KeyLightAzimuth = -35
    renderView1.KeyLightWarmth = 0.55  # > 0.5 warmer
    renderView1.FillLightKFRatio = fill_ratio  # lower = brighter fill
    renderView1.BackLightKBRatio = 3.5
    renderView1.HeadLightKHRatio = 3.0
    renderView1.MaintainLuminance = 1
    renderView1.UseToneMapping = 1
    clean_view(renderView1)

    view_from_x(renderView1)
    Render()
    save_png(RENDER_DIR / f"{out_name}.png", renderView1, RESOLUTION, SUPERSAMPLE)


# --------------------------------------- render --------------------------------------
for STL_IDX in STL_INDICES:
    STL_NAME = f"{STL_IDX:09}_abc"

    # STL
    reader = STLReader(FileNames=[str(STL_DIR / f"{STL_NAME}.stl")])
    reader.UpdatePipeline()
    render_and_save(reader, STL_NAME)
    Delete(reader)
    del reader

    # voxel VTI: keep only inside voxels (indicator in [1, 255]); 0 is background
    voxel = XMLImageDataReader(FileName=[str(VOXEL_DIR / f"{STL_NAME}.vti")])
    voxel.UpdatePipeline()

    threshold = Threshold(Input=voxel)
    threshold.Scalars = ["CELLS", "indicator"]
    threshold.LowerThreshold = 1
    threshold.UpperThreshold = 255
    threshold.UpdatePipeline()

    render_and_save(threshold, f"{STL_NAME}_voxel", metallic=0.1, key_intensity=1.0,
                    fill_ratio=4, color=(0.4, 0.5, 0.8))

    Delete(threshold)
    Delete(voxel)
    del threshold, voxel
