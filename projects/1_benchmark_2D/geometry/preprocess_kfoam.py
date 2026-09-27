import zipfile
from pathlib import Path

import numpy as np
import torch

from helper import box_mask, download, export, extract

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/kfoam").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/kfoam").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(9)

# -------------------------------------- settings -------------------------------------
URL = "https://zenodo.org/api/records/3532935/files/NMT_15_229_LLME_DivInterlayer%20%5B2015-10-09%2023.45.09%5D.zip/content"  # CC-BY 4.0
RESOLUTIONS = [128, 256]  # also possible: 512, 1024 inside the block
SAMPLES = 10  # per resolution, one volume
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the block
SHAPE = (1588, 1567, 1586)  # (z, y, x) from the .vgi header, 35.4 um voxels, uint8
KEEP_RAW = True  # keep the downloaded zip (projections + reconstruction) in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
file = RAW_DIR / "kfoam.zip"
if not file.exists():
    download(URL, file)

with zipfile.ZipFile(file) as archive:
    name = next(n for n in archive.namelist() if n.endswith(".raw"))
    volume = np.frombuffer(archive.read(name), dtype=np.uint8).reshape(SHAPE)

mask = box_mask(volume)  # open-cell foam block: the pores reach the surrounding air
clip = np.percentile(volume[mask][::7], CLIP)
geometries = {}
index = []
extract(volume, "graphite_foam", "kfoam", RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask)
print(f"kfoam: {volume.shape}, block fraction {mask.mean():.2f}")

if not KEEP_RAW:
    file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
