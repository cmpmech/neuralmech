import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from helper import download, export, extract, texture_mask

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/cfrp_t700").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/cfrp_t700").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(12)

# -------------------------------------- settings -------------------------------------
URL = "https://zenodo.org/api/records/7632124/files/data.zip/content"  # doi:10.5281/zenodo.7632124, CC-BY 4.0
SCANS = ["T700-T-02", "T700-T-08", "T700-T-21", "T700-T-26"]  # slow 0.4 um scans; the fast GF scans are noisy
RESOLUTIONS = [128, 256]  # the specimen is ~460 px wide, 512 does not fit
SAMPLES = 10  # per scan and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the fibre region
MARGIN = 10  # voxels (per 1000 px of slice width) kept clear of the fibre region edge
INVERT = True  # fibres are darker than the matrix in these reconstructions; invert so denser is brighter
KEEP_RAW = True  # keep the downloaded zip in RAW_DIR; the extracted h5 files are always removed

# ----------------------------------- preprocessing -----------------------------------
file = RAW_DIR / "data.zip"
if not file.exists():
    download(URL, file)

geometries = {}
index = []
with zipfile.ZipFile(file) as archive:
    for scan in SCANS:
        member = next(n for n in archive.namelist() if n.startswith(f"data/{scan}_pco_0p4um"))
        h5 = Path(archive.extract(member, RAW_DIR))  # h5py needs a seekable file
        with h5py.File(h5) as f:
            volume = f["data"][:].astype(np.float32)
        h5.unlink()
        if INVERT:
            volume = -volume

        # fibres run along axis 0, so axis-0 slices are cross-sections of the same footprint
        footprint = np.ones(volume.shape[1:], dtype=bool)
        for depth in (0.25, 0.5, 0.75):
            image = volume[int(depth * len(volume))]
            footprint &= texture_mask(image, np.ones(image.shape, dtype=bool), MARGIN)
        mask = np.broadcast_to(footprint, volume.shape)

        clip = np.percentile(volume[:, footprint][::5], CLIP)
        extract(volume, "cfrp_t700", scan, RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask, axes=(0,))
        print(f"{scan}: {volume.shape}, fibre region fraction {footprint.mean():.2f}")
        del volume, mask

if not KEEP_RAW:
    file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
