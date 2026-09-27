import io
import re
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import torch

from helper import download, export, extract, material_mask

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/syntactic_foam").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/syntactic_foam").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(10)

# -------------------------------------- settings -------------------------------------
RECORD = "https://zenodo.org/api/records/19699628/files"  # doi:10.5281/zenodo.19699628, CC-BY 4.0
SCANS = ["13_10"]  # 0.13 g/cm^3 hollow spheres at 10 vol%; 13_20 and 13_40 are deferred
RESOLUTIONS = [128, 256]  # also possible: 512, 1024 inside the cylinder
SAMPLES = 10  # per scan and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the specimen
KEEP_RAW = True  # keep the downloaded zips in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for scan in SCANS:
    file = RAW_DIR / f"{scan}_Rec.zip"
    if not file.exists():
        download(f"{RECORD}/{scan}_Rec.zip/content", file)

    with zipfile.ZipFile(file) as archive:
        # axial slices only; _RS2_Cor and _RS2_Sag are resliced copies of the same volume
        names = sorted(n for n in archive.namelist() if re.search(r"_rec\d+\.tif$", n))
        volume = np.stack([tifffile.imread(io.BytesIO(archive.read(n))) for n in names])

    mask = material_mask(volume)
    clip = np.percentile(volume[mask][::7], CLIP)
    extract(volume, f"syntactic_{scan}", scan, RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask)
    print(f"{scan}: {volume.shape}, specimen fraction {mask.mean():.2f}")
    del volume, mask

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
