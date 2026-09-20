import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
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

# ------------------------------------- load data -------------------------------------
fibers = np.load(DATA_DIR / f"fibers_anomaly_structured_{RESOLUTION}.npy")
data = torch.from_numpy(fibers).to(torch.float32)[None, None].to(device)

# ------------------------------------- load model ------------------------------------
model = torch.load(
    MODEL_DIR / f"fiber_ae_{BOTTLENECK_LAYERS}_{RESOLUTION}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# ----------------------------------- reconstruction ----------------------------------
with torch.no_grad():
    reconstruction = standardizex.inverse(model(standardizex(data))).cpu()
    error = (reconstruction - data.cpu()) ** 2

print(f"mse {torch.mean(error).item():.3e}")

# ----------------------------------- postprocessing ----------------------------------
fields = {
    "true": (data[0, 0].cpu(), dict(cmap="binary", vmin=0, vmax=1)),
    "pred": (reconstruction[0, 0], dict(cmap="binary", vmin=0, vmax=1)),
    "error": (error[0, 0], dict(cmap="hot_r", norm=LogNorm(vmin=1e-5, vmax=1.3))),
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
        fig.savefig(RGB_PDF_DIR / f"fibers_detection_{name}_structured.pdf")
    plt.close("all")
