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
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# model settings
KERNEL_SIZE, STRIDE, PADDING = 3, 1, 0

# hand-crafted 3x3 filters; in a CNN these weights would instead be learned
KERNELS = [
    ("identity", [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    ("shift and subtract", [[0, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ("edge detection", [[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]]),
    ("embossing", [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
    ("gaussian", (np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16).tolist()),
    ("averaging", (np.ones((3, 3)) / 9).tolist()),
    ("random", None),  # filled with random weights below
]

# ------------------------------------- load image ------------------------------------
img = Image.open(DATA_DIR / "images/water.jpg").convert("L")
x = torch.from_numpy(np.asarray(img)).unsqueeze(0).unsqueeze(0).to(torch.float32)

# ----------------------------------- set filters -------------------------------------
model = nn.Conv2d(1, 1, KERNEL_SIZE, STRIDE, PADDING, bias=False)
for i, (name, kernel) in enumerate(KERNELS):
    with torch.no_grad():
        if kernel is None:
            torch.manual_seed(1)
            model.weight[:] = torch.randn(model.weight.size())
        else:
            model.weight[:] = torch.tensor(kernel)

    with torch.no_grad():
        y = model(x)

# ----------------------------------- postprocessing ----------------------------------
    if not args.book:
        show_image(y[0, 0].numpy(), grayscale=True)
    else:
        show_image(
            y[0, 0].numpy(),
            grayscale=True,
            path=RGB_PDF_DIR / f"filter_example_{i}.pdf",
            close=True,
        )
