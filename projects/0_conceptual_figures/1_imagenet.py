import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
ids = {2, 3, 4, 9, 10, 12}

# ------------------------------------- load data -------------------------------------
ds = load_dataset("imagenet-1k", split="train", streaming=True)
for i, sample in enumerate(ds):
    if i in ids:
        img = np.array(sample["image"])
        label = ds.features["label"].int2str(int(sample["label"]))
# ----------------------------------- postprocessing ----------------------------------
        print(label)
        show_image(img, path=RESULTS_DIR / f"imagenet_{i}.png", close=args.book)
        ids.discard(i)
    if not ids:
        break

del ds, sample
os._exit(0)
