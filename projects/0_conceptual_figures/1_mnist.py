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

ds = load_dataset("ylecun/mnist", split="train", streaming=True)
seen = set()
for sample in ds:
    digit = sample["label"]
    if digit in seen:
        continue
    seen.add(digit)
    img = 255 - np.array(sample["image"])  # invert color scheme
    print(digit)
    show_image(
        img, grayscale=True, path=RESULTS_DIR / f"mnist_{digit}.png", close=args.book
    )
    if len(seen) == 10:
        break

del ds, sample
os._exit(0)
