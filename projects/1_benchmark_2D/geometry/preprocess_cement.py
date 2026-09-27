import io
from pathlib import Path

import libarchive
import numpy as np
import tifffile
import torch

from helper import download, export, extract, material_mask

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/cement").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/cement").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(8)

# -------------------------------------- settings -------------------------------------
RECORD = "https://zenodo.org/api/records/2533863/files"  # doi:10.5281/zenodo.2533863, CC-BY 4.0
PASTES = {"PC": "portland", "pc-cc": "portland_calcite", "PC-FA": "portland_flyash"}
RECONSTRUCTION = "TIFF_delta"  # electron density; TIFF_beta is the weaker attenuation contrast
RESOLUTIONS = [128, 256]  # also possible: 512 inside the pillar
SAMPLES = 10  # per paste and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the specimen
KEEP_RAW = True  # keep the downloaded archives in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for name, kind in PASTES.items():
    file = RAW_DIR / f"{name}.rar"
    if not file.exists():
        download(f"{RECORD}/{name}.rar/content", file)

    slices = {}
    with libarchive.file_reader(str(file)) as archive:
        for entry in archive:
            if entry.pathname.startswith(RECONSTRUCTION) and entry.pathname.endswith(".tif"):
                slices[entry.pathname] = tifffile.imread(io.BytesIO(b"".join(entry.get_blocks())))
    volume = np.stack([slices[k] for k in sorted(slices)])
    del slices

    mask = material_mask(volume)
    clip = np.percentile(volume[mask][::7], CLIP)
    extract(volume, kind, name, RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask)
    print(f"{name}: {volume.shape}, specimen fraction {mask.mean():.2f}")

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
