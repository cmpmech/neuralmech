import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
SAMPLES = {
    "hawksbill": 4,
    "trilobite": 7,
    "okapi": 4,
    "sunflower": 2,
    "euphonium": 3,
    "flamingo": 4,
}  # category: image number
ASPECT = 4 / 3

# ------------------------------------- load data -------------------------------------
archive = DATA_DIR / "caltech-101.zip"
image_dir = DATA_DIR / "101_ObjectCategories"
if not image_dir.exists():
    if not archive.exists():
        # caltechdata refuses python's default user agent
        request = urllib.request.Request(URL, headers={"User-Agent": "neuralmech"})
        with urllib.request.urlopen(request) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
    with zipfile.ZipFile(archive) as zf:
        inner = zf.open("caltech-101/101_ObjectCategories.tar.gz")
        with tarfile.open(fileobj=inner, mode="r|gz") as tf:
            tf.extractall(DATA_DIR, filter="data")

for category, number in SAMPLES.items():
    path = image_dir / category / f"image_{number:04d}.jpg"
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    crop_h = min(h, round(w / ASPECT))
    crop_w = min(w, round(h * ASPECT))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    img = img[top : top + crop_h, left : left + crop_w]
# ----------------------------------- postprocessing ----------------------------------
    show_image(img, path=RGB_PDF_DIR / f"caltech101_{category}.pdf", close=args.book)
