from pathlib import Path

import h5py
import numpy as np
import torch

from helper import dpmp_files, dpmp_link, download, export, extract, percentile_clip

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/mrccm").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/mrccm").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(2)

# -------------------------------------- settings -------------------------------------
PROJECT = 362  # DPMP project, doi:10.17612/3t36-q704, ODC-BY 1.0
RESOLUTIONS = [128, 256]  # also possible: 512, 1024
SAMPLES = 10  # per volume and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255
KEEP_RAW = True  # keep the downloaded volumes in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for sample, path in dpmp_files(PROJECT):
    if "HR (.mat) volumes" not in path:
        continue
    file = RAW_DIR / Path(path).name
    if not file.exists():
        download(dpmp_link(PROJECT, path), file)

    try:
        with h5py.File(file) as f:
            key = next(k for k in f if not k.startswith("#"))
            volume = f[key][:]
    except OSError as error:
        print(f"skip {file.name}: {error}")  # ILS4.mat is truncated on the server
        volume = None

    if volume is not None:
        clip = percentile_clip(volume, CLIP)
        kind = sample.split("(")[1].rstrip(")")
        extract(volume, kind, file.stem, RESOLUTIONS, SAMPLES, rng, geometries, index, clip)
        print(f"{sample}: {volume.shape}")

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
