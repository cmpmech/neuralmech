import io
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy import ndimage

from helper import download, export, extract, texture_mask

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/gfrp_ud").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/gfrp_ud").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(11)

# -------------------------------------- settings -------------------------------------
RECORD = "https://zenodo.org/api/records/1195879/files"  # doi:10.5281/zenodo.1195879, CC-BY 4.0
SCANS = {"XCT_L": "XCT_LR/", "XCT_M": "XCT_MR/", "XCT_H": "XCT_HR/", "SRCT": "SCT/"}  # zip -> folder
RESOLUTIONS = [128, 256]  # also possible: 512 (lab CT), 1024 (synchrotron)
SAMPLES = 10  # per scan and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the specimen
MARGIN = 10  # voxels (per 1000 px of slice width) kept clear of the fibre region edge
KEEP_RAW = True  # keep the downloaded zips in RAW_DIR

# --------------------------------------- helper --------------------------------------
def field_of_view(image, synchrotron):
    if synchrotron:  # mid-gray background around a tilted specimen
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]
        return (y - image.shape[0] / 2) ** 2 + (x - image.shape[1] / 2) ** 2 < (0.47 * image.shape[0]) ** 2
    scale = image.shape[0] / 1000  # circular lab-CT field of view, zero outside; its bright rim is excluded
    fov = ndimage.gaussian_filter(image, 4 * scale) > 0.05 * image.max()
    return ndimage.binary_erosion(fov, iterations=int(30 * scale))


# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for scan, folder in SCANS.items():
    file = RAW_DIR / f"{scan}.zip"
    if not file.exists():
        download(f"{RECORD}/{scan}.zip/content", file)

    with zipfile.ZipFile(file) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith(folder) and n.endswith(".tif"))
        images = [tifffile.imread(io.BytesIO(archive.read(n))) for n in names]
    shape = max(set(i.shape for i in images), key=[i.shape for i in images].count)
    volume = np.stack([i for i in images if i.shape == shape])  # XCT_M holds one stray transposed slice
    del images

    # fibres run (slightly inclined) along the stack axis, so every slice is a cross-section;
    # the specimen is where the fibre texture is, intersected over three depths
    footprint = np.ones(volume.shape[1:], dtype=bool)
    for depth in (0.25, 0.5, 0.75):
        image = volume[int(depth * len(volume))].astype(np.float32)
        footprint &= texture_mask(image, field_of_view(image, scan == "SRCT"), MARGIN)
    mask = np.broadcast_to(footprint, volume.shape)

    clip = np.percentile(volume[:, footprint][::3], CLIP)
    kind = f"gfrp_{scan.lower()}"
    extract(volume, kind, scan, RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask, axes=(0,))
    print(f"{scan}: {volume.shape}, specimen fraction {footprint.mean():.2f}")

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
