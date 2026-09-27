from pathlib import Path

import h5py
import numpy as np
import torch

from helper import dpmp_files, dpmp_link, download, export, extract, percentile_clip, type_name

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/presalt").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/presalt").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(1)

# -------------------------------------- settings -------------------------------------
PROJECT = 503  # DPMP project, doi:10.17612/xr50-s717, ODC-BY 1.0
RESOLUTIONS = [128, 256]  # also possible: 512, 1024
SAMPLES = 10  # per volume and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255
KEEP_RAW = True  # keep the downloaded volumes in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for sample, path in dpmp_files(PROJECT):
    if "high resolution (grayscale)" not in path or not path.endswith(".nc"):
        continue
    file = RAW_DIR / Path(path).name
    if not file.exists():
        download(dpmp_link(PROJECT, path), file)

    with h5py.File(file) as f:
        volume = f["data"][:]
    clip = percentile_clip(volume, CLIP)
    extract(volume, type_name(sample), file.stem, RESOLUTIONS, SAMPLES, rng, geometries, index, clip)
    print(f"{sample}: {volume.shape}")

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
