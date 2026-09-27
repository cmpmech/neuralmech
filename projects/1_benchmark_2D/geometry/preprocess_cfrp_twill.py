from pathlib import Path

import numpy as np
import torch

from helper import download, export, extract, material_mask

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/cfrp_twill").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/cfrp_twill").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(6)

# -------------------------------------- settings -------------------------------------
RECORD = "https://zenodo.org/api/records/11200931/files"  # doi:10.5281/zenodo.11200931, CC-BY 4.0
RESOLUTIONS = [128, 256]  # also possible: 512 in the plane of the plate
SAMPLES = 10  # per volume and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the specimen
KEEP_RAW = True  # keep the downloaded volumes in RAW_DIR

# (z, y, x) from "Data Volume Sizes.txt", 15 um voxels; P1 and P3 carry trailing noise after the volume
VOLUMES = {
    "P1-Random": (900, 600, 1000),
    "P3-Random": (900, 400, 1000),
    "P5-Controlled": (900, 560, 1038),
}

# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for name, shape in VOLUMES.items():
    file = RAW_DIR / f"{name}.raw"
    if not file.exists():
        download(f"{RECORD}/{name}.raw/content", file)

    volume = np.fromfile(file, dtype=np.uint8, count=int(np.prod(shape))).reshape(shape)
    mask = material_mask(volume)
    clip = np.percentile(volume[mask][::7], CLIP)
    extract(volume, "cfrp_twill", name, RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask)
    print(f"{name}: {volume.shape}, specimen fraction {mask.mean():.2f}")

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
