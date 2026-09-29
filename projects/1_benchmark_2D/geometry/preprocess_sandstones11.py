from pathlib import Path

import numpy as np
import torch

from helper import SETTINGS, download, dpmp_files, dpmp_link, export, extract, percentile_clip, type_name

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/sandstones11").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/sandstones11").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
PROJECT = 317  # DPMP project, doi:10.17612/f4h1-w124, ODC-BY 1.0
KEEP_RAW = True  # keep the downloaded volumes in RAW_DIR
SHAPE = (1000, 1000, 1000)

# ----------------------------------- preprocessing -----------------------------------
geometries = {}
index = []
for sample, path in dpmp_files(PROJECT):
    if not path.endswith("_grayscale.raw"):
        continue
    file = RAW_DIR / Path(path).name
    if not file.exists():
        download(dpmp_link(PROJECT, path), file)

    volume = np.fromfile(file, dtype=np.uint8).reshape(SHAPE)
    clip = percentile_clip(volume, SETTINGS["clip"])
    extract(volume, type_name(sample), file.stem, rng, geometries, index, clip)
    print(f"{sample}: {volume.shape}")

    if not KEEP_RAW:
        file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
