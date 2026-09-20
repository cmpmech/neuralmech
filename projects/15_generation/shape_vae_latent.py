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
BETA = 0.05  # 0.0 reconstructs best, 0.2 gives the tidiest latent space
SAMPLING_STEPS = 5  # draws per shape from the encoded distribution
SAMPLESX = 6
SAMPLESY = 6

# ------------------------------------- load data -------------------------------------
data = []
for label in LABELS:
    shapes = np.load(DATA_DIR / f"shapes_{label}_{DOMAIN_SIZE}.npy")
    data.append(torch.from_numpy(shapes).to(torch.float32).unsqueeze(1).to(device))

# ------------------------------------- load model ------------------------------------
model = torch.load(
    MODEL_DIR / f"shape_vae_2_{BETA}_{DOMAIN_SIZE}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# --------------------------------- latent space walk ---------------------------------
latents = []
with torch.no_grad():
    for shape in data:
        latent = []
        for _ in range(SAMPLING_STEPS):
            distributions = model.encode(standardizex(shape))
            mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
            std = torch.exp(0.5 * logvar)
            latent.append(mean + torch.randn_like(std) * std)
        latents.append(torch.cat(latent, 0).cpu())

x = torch.linspace(-1.0, 1.0, SAMPLESX)
y = torch.linspace(-1.0, 1.0, SAMPLESY)
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

    fig, ax = plt.subplots(SAMPLESY, SAMPLESX, figsize=(SAMPLESX, SAMPLESY))
    for i in range(SAMPLESX):
        for j in range(SAMPLESY):
            ax[j, i].imshow(
                gen_shapes[i, j].T, cmap="binary", origin="lower", vmin=0, vmax=1
            )
            ax[j, i].set_aspect("equal")
            ax[j, i].axis("off")
            ax[j, i].set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for label, latent in zip(LABELS, latents):
        save_csv(
            CSV_DIR / f"shapes_vae_latent_{BETA}_{label}.csv",
            x=latent[:, 0],
            y=latent[:, 1],
        )

    for i in range(SAMPLESX):
        for j in range(SAMPLESY):
            fig, ax = plt.subplots(figsize=(1, 1), dpi=DOMAIN_SIZE)
            ax.imshow(gen_shapes[i, j].T, cmap="binary", origin="lower", vmin=0, vmax=1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_rasterized(True)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(RGB_PDF_DIR / f"genshapes_vae_{BETA}_{i}{j}.pdf")
            plt.close(fig)
