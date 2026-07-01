# run with pvpython
from pathlib import Path

from paraview.simple import *
from pvpython_helper import clean_view, save_png, view_from_x

BASE_DIR = Path(__file__).parent

# ----------------------------------- settings ----------------------------------------
NAME = "WD-150"  # B-HAI-1, KAK-1, KAK-2, BM-5, BM-48, BM-105, WD-2, WD-150
RESOLUTION = [500, 1000]

# --------------------------------------- render --------------------------------------
reader = XMLRectilinearGridReader(
    FileName=[str(BASE_DIR / f"../../results/3D/{NAME}.vtr")]
)
reader.UpdatePipeline()

display = Show(reader)
display.SetRepresentationType("Surface")
ColorBy(display, ("CELLS", "scalar"))

renderView1 = GetActiveViewOrCreate("RenderView")
scalarLUT = GetColorTransferFunction("scalar")
scalarLUT.ApplyPreset("X Ray", True)
HideScalarBarIfNotNeeded(scalarLUT, renderView1)
clean_view(renderView1)

view_from_x(renderView1)
Render()

save_png(BASE_DIR / f"../../results/rgb_png/rocks_{NAME}.png", renderView1, RESOLUTION)
