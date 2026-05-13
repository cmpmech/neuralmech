import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

ids = [2, 3, 4, 9, 10, 12]

ds = load_dataset("imagenet-1k", split="train", streaming=True)
n = 20
for i, sample in enumerate(ds):
    img = np.array(sample["image"])
    label = ds.features["label"].int2str(sample["label"])
    if i in ids:
        print(label)
        show_image(img, path=RESULTS_DIR / f"imagenet_{i}.png", close=args.book)
    if i >= n - 1:
        break

del ds, sample
os._exit(0)
