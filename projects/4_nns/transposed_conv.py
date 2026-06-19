import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# model settings
# kernel size not divisible by the stride causes checkerboarding
KERNEL_SIZE, STRIDE, PADDING = 3, 2, 0

# ------------------------------------- load image ------------------------------------
img = np.asarray(Image.open(DATA_DIR / "images/onions.jpg").convert("L")).copy()
x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32)

model = nn.ConvTranspose2d(1, 1, KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=False)
with torch.no_grad():
    model.weight.fill_(1.0)
    y = model(x)

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    show_image(x[0, 0].numpy(), grayscale=True)
    show_image(y[0, 0].numpy(), grayscale=True)

# -------------------------------- book postprocessing --------------------------------
else:
    show_image(x[0, 0].numpy(), grayscale=True, path=RESULTS_DIR / "checkerboarding1.jpg", close=True)
    show_image(y[0, 0].numpy(), grayscale=True, path=RESULTS_DIR / "checkerboarding2.jpg", close=True)
