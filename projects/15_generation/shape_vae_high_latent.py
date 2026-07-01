import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- load trained model --------------------------
DOMAIN_SIZE = 128
LATENT_DIM = 32
# LATENT_DIM = 2

model = torch.load(
    BASE_DIR / f"../../models/shape_vae_{LATENT_DIM}_{0.05}_{DOMAIN_SIZE}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# ------------------------ latent space sampling -------------------------
samples = 1000 if args.animate else 15
scale = 2.0 if args.animate else 0.3

start = scale * torch.ones((1, LATENT_DIM), dtype=torch.float32, device=device)
end = -scale * torch.ones((1, LATENT_DIM), dtype=torch.float32, device=device)
t = torch.linspace(0, 1, samples).unsqueeze(1).to(device)
z = start + (end - start) * t

with torch.no_grad():
    gen_shapes = standardizex.inverse(model.decode(z))
gen_shapes = gen_shapes.squeeze().cpu()

# ---------------------------- postprocessing ----------------------------
if not args.book and not args.animate:
    fig, ax = plt.subplots(1, samples, figsize=(samples, 1), dpi=DOMAIN_SIZE)
    for i in range(samples):
        ax[i].imshow(gen_shapes[i].T, cmap="binary", vmin=0, vmax=1, origin="lower")
        ax[i].axis("off")
    plt.show()

if args.book:
    for i in range(samples):
        fig, ax = plt.subplots(figsize=(1, 1), dpi=DOMAIN_SIZE)
        ax.imshow(gen_shapes[i].T, cmap="binary", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(
            RGB_PDF_DIR / f"genshapes_vae_high_{LATENT_DIM}_{i}.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

if args.animate:
    folder = ANIMATION_DIR / f"vae_high_latent_{LATENT_DIM}"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(samples):
        fig, ax = plt.subplots(figsize=(1, 1), dpi=DOMAIN_SIZE)
        ax.imshow(gen_shapes[i].T, cmap="binary", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(
            folder / f"frame_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
