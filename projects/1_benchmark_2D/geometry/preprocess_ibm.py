import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import torch

from helper import SETTINGS, download, export, extract, percentile_clip

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/ibm").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/ibm").resolve()
PART_DIR = (GEOMETRY_DIR / "parts").resolve()  # crops per plug, so an interrupted run resumes
PART_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
ARTICLE = 21375565  # Figshare+ doi:10.25452/figshare.plus.21375565, CDLA-Sharing-1.0
ROI = "grayscale_ROI-1"  # one of three 2500^3 regions per plug; _filtered_ and binary_ also exist
KEEP_RAW = False  # keep the downloaded archives in RAW_DIR, 3-8 GB each (95 GB total)
SHAPE = (2500, 2500, 2500)

# ----------------------------------- preprocessing -----------------------------------
with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{ARTICLE}") as response:
    files = json.load(response)["files"]

geometries = {}
index = []
entries = [e for e in files if e["name"].endswith(f"_{ROI}.raw.tar.bz2")]
for i, entry in enumerate(entries):
    kind = entry["name"].split("_2p25um")[0].replace("-", "_")
    name = entry["name"].split(".")[0]
    part = PART_DIR / f"{name}.pt"
    if not part.exists():
        file = RAW_DIR / entry["name"]
        if not file.exists():
            download(entry["download_url"], file)

        command = ["tar", "-xjOf", str(file), "--exclude=._*"]  # some archives also carry macOS ._ files
        raw = subprocess.run(command, capture_output=True, check=True).stdout
        volume = np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE)
        clip = percentile_clip(volume, SETTINGS["clip"], stride=8)
        rng = np.random.default_rng([4, i])  # seeded per plug, so resuming reproduces the same crops
        part_geometries, part_index = {}, []
        extract(volume, kind, name, rng, part_geometries, part_index, clip)
        torch.save((part_geometries, part_index), part)
        print(f"{kind}: {volume.shape}")

        del raw, volume
        if not KEEP_RAW:
            file.unlink()

    part_geometries, part_index = torch.load(part, weights_only=False)
    for key, images in part_geometries.items():
        geometries.setdefault(key, []).extend(images)
    index.extend(part_index)

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
