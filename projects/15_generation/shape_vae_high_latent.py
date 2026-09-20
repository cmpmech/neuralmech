import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DOMAIN_SIZE = 128
LATENT_DIM = 32  # 2 reproduces the plane the latent drivers walk over
BETA = 0.05

# a diagonal of the latent cube, densely resampled for the animation
SAMPLES = 1000 if args.animate else 15
SCALE = 2.0 if args.animate else 0.3

ANIMATION_DIR = (RESULTS_DIR / f"animations/vae_high_latent_{LATENT_DIM}").resolve()

# ------------------------------------- load model ------------------------------------
model = torch.load(
    MODEL_DIR / f"shape_vae_{LATENT_DIM}_{BETA}_{DOMAIN_SIZE}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# --------------------------------- latent space walk ---------------------------------
start = SCALE * torch.ones((1, LATENT_DIM), dtype=torch.float32, device=device)
end = -SCALE * torch.ones((1, LATENT_DIM), dtype=torch.float32, device=device)
t = torch.linspace(0, 1, SAMPLES).unsqueeze(1).to(device)
z = start + (end - start) * t

with torch.no_grad():
    gen_shapes = standardizex.inverse(model.decode(z))
gen_shapes = gen_shapes.squeeze().cpu()

# ----------------------------------- postprocessing ----------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

if not args.book and not args.animate:
    fig, ax = plt.subplots(1, SAMPLES, figsize=(SAMPLES, 1), dpi=DOMAIN_SIZE)
    for i in range(SAMPLES):
        ax[i].imshow(gen_shapes[i].T, cmap="binary", vmin=0, vmax=1, origin="lower")
        ax[i].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for i in range(SAMPLES):
        fig, ax = plt.subplots(figsize=(1, 1), dpi=DOMAIN_SIZE)
        ax.imshow(gen_shapes[i].T, cmap="binary", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if args.animate:
            plt.savefig(ANIMATION_DIR / f"frame_{i}.jpg")
        else:
            plt.savefig(RGB_PDF_DIR / f"genshapes_vae_high_{LATENT_DIM}_{i}.pdf")
        plt.close(fig)
