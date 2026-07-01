import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
# TODO could be animated
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 128
EXAMPLES = 7

# ------------------------------------- load data -------------------------------------
data = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fibers_test_{RESOLUTION}.npy")
)
masks = torch.from_numpy(
    np.load(BASE_DIR / f"../../data/graded_fiber_masks_test_{RESOLUTION}.npy")
)
data = data.to(torch.float32)
masks = masks.to(torch.float32)

x_gt = data[:EXAMPLES].unsqueeze(1).to(device)  # (N, 1, H, W)
masks = masks[:EXAMPLES].unsqueeze(1).to(device)  # (N, 1, H, W)
b = x_gt * masks

# --------------------------------------- model ---------------------------------------
model = torch.load(
    BASE_DIR / f"../../models/modl_{RESOLUTION}.pt2",
    map_location=device,
    weights_only=False,
)
model.eval()

# ----------------------------------- reconstruction ----------------------------------
with torch.no_grad():
    x_rec = model(b, masks)

# ----------------------------------- postprocessing ----------------------------------
for i in range(EXAMPLES):
    fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax.imshow(x_rec[i, 0].T.cpu(), origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PDF_DIR / f"img_prediction_modl_{i}.pdf")
    plt.close()
