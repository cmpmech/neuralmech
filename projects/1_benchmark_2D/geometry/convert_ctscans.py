import csv
from pathlib import Path

import numpy as np
import tifffile

from helper import SETTINGS

BASE_DIR = Path(__file__).parent
EXT_DATA_DIR = (BASE_DIR / "../../../external_data").resolve()
SCAN_DIR = (EXT_DATA_DIR / "ctscans").resolve()
SCAN_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------- settings -------------------------------------
OVERWRITE = False  # redo scans that already exist in SCAN_DIR

# ----------------------------------- preprocessing -----------------------------------
with open(EXT_DATA_DIR / "ct_scans.csv") as f:
    scans = list(csv.DictReader(f))

for scan in scans:
    out = SCAN_DIR / f"{scan['name']}.npy"
    if out.exists() and not OVERWRITE:
        continue
    files = sorted((EXT_DATA_DIR / scan["source"]).glob("*.tif"))  # external_data/TriaxCTScans links to the originals
    shape = tuple(int(scan[k]) for k in ("nx", "ny", "nz"))
    binning = len(files) // shape[2]  # the 2000^2 vg-data stacks are binned 3x in every direction

    volume = np.empty(shape, dtype=np.float32)
    for z in range(shape[2]):
        stack = np.stack([tifffile.imread(f) for f in files[z * binning : (z + 1) * binning]])
        image = stack.astype(np.float32).mean(axis=0)[: shape[0] * binning, : shape[1] * binning]
        volume[:, :, z] = image.reshape(shape[0], binning, shape[1], binning).mean(axis=(1, 3))
    if scan["z_reversed"] == "1":
        volume = volume[:, :, ::-1]

    sample = volume[::4, ::4, ::4]
    air, rock = np.percentile(sample, [1, 99])
    lo, hi = np.percentile(sample[sample > (air + rock) / 2], SETTINGS["clip"])
    volume = np.round(255 * np.clip((volume - lo) / (hi - lo), 0, 1)).astype(np.uint8)

    np.save(out, volume)
    print(f"{scan['name']}: {len(files)} tifs, binning {binning}, gray range {lo:.0f}-{hi:.0f}, saved {out.name}")
