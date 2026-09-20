import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 256
BOTTLENECK_LAYERS = 1
ANOMALY_DEGREES = 11  # number of squares replacing fibers, 0 (none) to 10 (all)
DEGREE = 0  # the degree shown in the book figures
SAMPLE = 0

# ------------------------------------- load data -------------------------------------
data = []
for i in range(ANOMALY_DEGREES):
    fibers = np.load(DATA_DIR / f"fibers_anomaly_{i}_{RESOLUTION}.npy")
    data.append(torch.from_numpy(fibers).to(torch.float32).unsqueeze(1).to(device))

# ------------------------------------- load model ------------------------------------
model = torch.load(
    MODEL_DIR / f"fiber_ae_{BOTTLENECK_LAYERS}_{RESOLUTION}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# ----------------------------------- reconstruction ----------------------------------
reconstructions = []
errors = []
with torch.no_grad():
    for fibers in data:
        reconstructions.append(standardizex.inverse(model(standardizex(fibers))).cpu())
        errors.append((reconstructions[-1] - fibers.cpu()) ** 2)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
for error in errors:
    ax.hist(torch.mean(error, dim=(1, 2, 3)), bins=50)

fields = {
    "true": (data[DEGREE][SAMPLE][0].cpu(), dict(cmap="binary", vmin=0, vmax=1)),
    "pred": (reconstructions[DEGREE][SAMPLE][0], dict(cmap="binary", vmin=0, vmax=1)),
    "error": (
        errors[DEGREE][SAMPLE][0],
        dict(cmap="hot_r", norm=LogNorm(vmin=1e-5, vmax=1.3)),
    ),
}
figures = {}
for name, (field, style) in fields.items():
    figures[name], ax = plt.subplots(figsize=(1, 1), dpi=RESOLUTION)
    ax.imshow(field.T, origin="lower", **style)
    ax.axis("off")
    figures[name].subplots_adjust(left=0, right=1, top=1, bottom=0)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, fig in figures.items():
        fig.savefig(RGB_PDF_DIR / f"fibers_detection_{name}_{DEGREE}.pdf")
    plt.close("all")

    for i, error in enumerate(errors):
        save_csv(
            CSV_DIR / f"fibers_mean_error_{i}.csv",
            x=torch.arange(1, error.shape[0] + 1),
            y=torch.mean(error, dim=(1, 2, 3)),
        )
