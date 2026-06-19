import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/images").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(2)

# -------------------------------------- settings -------------------------------------
STEPS = 100

# ------------------------------------- load image ------------------------------------
img = Image.open(DATA_DIR / "weimar_small.jpg").convert("RGB")
img = np.asarray(img) / 255 * 2 - 1

# -------------------------------------- noising --------------------------------------
beta = lambda t: (t - 1) / STEPS  # starting at 1
alpha = lambda t: 1 - beta(t)

for t in [1, 10, 15, 100]:
    eff_alpha = np.prod([alpha(s) for s in range(1, t + 1)])

    noise = rng.normal(0, 1, img.shape)
    noisy_img = np.sqrt(eff_alpha) * img + np.sqrt(1 - eff_alpha) * noise

# ----------------------------------- postprocessing ----------------------------------
    show_image(
        (noisy_img + 1) / 2,
        path=RESULTS_DIR / f"diffusion_step_{t}.png",
        close=args.book,
    )
