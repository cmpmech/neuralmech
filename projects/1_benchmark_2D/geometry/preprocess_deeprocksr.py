import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from helper import SETTINGS, download, dpmp_link, export, normalize, shape_name

BASE_DIR = Path(__file__).parent
RAW_DIR = (BASE_DIR / "../../../external_data/2D_benchmark/deeprocksr").resolve()
GEOMETRY_DIR = (BASE_DIR / "../../../data/2D_benchmark/geometries/deeprocksr").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
rng = np.random.default_rng(5)
rect_rng = rng.spawn(1)[0]  # rectangles draw separately, so the squares stay the same

# -------------------------------------- settings -------------------------------------
PROJECT = 215  # DPMP project, doi:10.17612/s3m9-e024, ODC-BY 1.0
PATH = "2D Dataset/DeepRock-SR-2D/DeepRockSR-2D.zip"
KINDS = ["sandstone", "carbonate", "coal"]  # shuffled2D only mixes these three
KEEP_RAW = True  # keep the downloaded zip in RAW_DIR

# ----------------------------------- preprocessing -----------------------------------
file = RAW_DIR / Path(PATH).name
if not file.exists():
    download(dpmp_link(PROJECT, PATH), file)

geometries = {}
index = []
with zipfile.ZipFile(file) as archive:
    for kind in KINDS:
        names = [n for n in archive.namelist() if f"/{kind}2D_" in n and "_HR/" in n]
        names = sorted(n for n in names if n.endswith(".png"))
        images = [np.array(Image.open(io.BytesIO(archive.read(n))).convert("L")) for n in names]
        clip = np.percentile(np.stack(images[::20]), SETTINGS["clip"])
        print(f"{kind}: {len(images)} images, clip {clip}")

        for aspect in SETTINGS["aspect_ratios"]:
            generator = rng if aspect == 1 else rect_rng
            for res in SETTINGS["resolutions"]:
                crops = geometries.setdefault((kind, shape_name(res, aspect)), [])
                for idx in generator.choice(len(names), SETTINGS["samples"], replace=False):
                    image = images[idx]
                    # a rectangle is cut along either direction and transposed back
                    transposed = aspect != 1 and bool(generator.random() < 0.5)
                    size = (res // aspect, res) if transposed else (res, res // aspect)
                    x0 = int(generator.integers(image.shape[0] - size[0] + 1))
                    y0 = int(generator.integers(image.shape[1] - size[1] + 1))
                    image = normalize(image[x0 : x0 + size[0], y0 : y0 + size[1]], clip)
                    index.append([f"{kind}_{shape_name(res, aspect)}_{len(crops)}", names[idx], 2, 0, x0, y0, transposed])
                    crops.append(image.T if transposed else image)

if not KEEP_RAW:
    file.unlink()

# --------------------------------------- export --------------------------------------
export(geometries, index, GEOMETRY_DIR)
