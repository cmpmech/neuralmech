import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(2)

# ------------------------------ load image ------------------------------
img = Image.open(DATA_DIR / "images/weimar_small.jpg").convert("RGB")
img = np.asarray(img) / 255 * 2 - 1
# ------------------------------- noising --------------------------------
STEPS = 100

beta = lambda t: (t - 1) / STEPS  # starting at 0?
alpha = lambda t: 1 - beta(t)

for t in [0, 10, 15, 100]:
    eff_alpha = np.prod([alpha(s) for s in range(t)])

    noise = rng.normal(0, 1, img.shape)
    noisy_img = np.sqrt(eff_alpha) * img + np.sqrt(1 - eff_alpha) * noise

    print(np.min(img))
    print(np.max(img))

# ------------------------------- original -------------------------------
    show_image(
        (noisy_img + 1) / 2,
        path=RESULTS_DIR / f"diffusion_step_{t}.png",
        close=args.book,
    )
