import zipfile
from pathlib import Path

import numpy as np
import torch

from helper import box_mask, download, export, extract, read_dicom

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/alsi10mg").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/alsi10mg").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(7)

# -------------------------------------- settings -------------------------------------
URL = "https://zenodo.org/api/records/17602110/files/Tomographic%20image%20stacks.zip/content"  # CC-BY 4.0
RESOLUTIONS = [128, 256]  # also possible: 512 in G1 and G2
SAMPLES = 10  # per volume and resolution
CLIP = (0.1, 99.9)  # grayscale percentiles mapped to 0 and 255, inside the lattice
MARGIN = 0.05  # fraction of the lattice bounding box trimmed on each side
KEEP_RAW = True  # keep the downloaded zip in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
file = RAW_DIR / "stacks.zip"
if not file.exists():
    download(URL, file)

geometries = {}
index = []
with zipfile.ZipFile(file) as archive:
    for name in ["G1", "G2", "GD"]:
        slices = sorted(n for n in archive.namelist() if f"/{name}/" in n and n.endswith(".dcm"))
        volume = np.stack([read_dicom(archive.read(n)) for n in slices])

        mask = box_mask(volume, MARGIN)  # the air between struts is geometry, the air around it is not
        clip = np.percentile(volume[mask][::7], CLIP)
        extract(volume, "alsi10mg", name, RESOLUTIONS, SAMPLES, rng, geometries, index, clip, mask)
        print(f"{name}: {volume.shape}, lattice fraction {mask.mean():.2f}")

if not KEEP_RAW:
    file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
