import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

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
DOMAIN_SIZE = 128
LABELS = ["circle", "ellipse", "square", "triangle", "cross", "star"]

# the two windows of the latent plane the book walks over
BOX = 1
BOXES = {
    1: (10, 6, (-0.9, 0.3), (-0.2, 0.6)),
    2: (6, 8, (0.1, 0.6), (0.6, 1.5)),
}

# ------------------------------------- load data -------------------------------------
data = []
for label in LABELS:
    shapes = np.load(DATA_DIR / f"shapes_{label}_{DOMAIN_SIZE}.npy")
    data.append(torch.from_numpy(shapes).to(torch.float32).unsqueeze(1).to(device))

# ------------------------------------- load model ------------------------------------
model = torch.load(
    MODEL_DIR / f"shape_ae_{DOMAIN_SIZE}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# --------------------------------- latent space walk ---------------------------------
latents = []
with torch.no_grad():
    for shape in data:
        latents.append(model.encode(standardizex(shape)).cpu())

samplesx, samplesy, x_range, y_range = BOXES[BOX]
x = torch.linspace(*x_range, samplesx)
y = torch.linspace(*y_range, samplesy)
x, y = torch.meshgrid(x, y, indexing="ij")
z = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1).to(device)

with torch.no_grad():
    gen_shapes = standardizex.inverse(model.decode(z))
gen_shapes = gen_shapes.reshape(*x.shape, DOMAIN_SIZE, DOMAIN_SIZE).cpu()

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    colors = cm.tab10(np.linspace(0, 1, len(latents)))

    fig, ax = plt.subplots()
    for i, latent in enumerate(latents):
        ax.plot(latent[:, 0], latent[:, 1], "o", color=colors[i])
    ax.plot(x.flatten(), y.flatten(), "k.")
    ax.set_aspect("equal")

    fig, ax = plt.subplots(samplesy, samplesx, figsize=(samplesx, samplesy))
    for i in range(samplesx):
        for j in range(samplesy):
            ax[j, i].imshow(gen_shapes[i, j].T, cmap="binary", origin="lower")
            ax[j, i].set_aspect("equal")
            ax[j, i].axis("off")
            ax[j, i].set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for label, latent in zip(LABELS, latents):
        save_csv(
            CSV_DIR / f"shapes_ae_latent_{label}.csv",
            x=latent[:, 0],
            y=latent[:, 1],
        )

    for i in range(samplesx):
        for j in range(samplesy):
            fig, ax = plt.subplots(figsize=(1, 1), dpi=DOMAIN_SIZE)
            ax.imshow(gen_shapes[i, j].T, cmap="binary", origin="lower", vmin=0, vmax=1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_rasterized(True)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(RGB_PDF_DIR / f"genshapes_ae_{i}{j}_{BOX}.pdf")
            plt.close(fig)
