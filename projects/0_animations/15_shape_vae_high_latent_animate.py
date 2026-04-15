from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- load trained model --------------------------
domain_size = 128
# latent_dim = 2
latent_dim = 32

model = torch.load(
    BASE_DIR / f"../../models/shape_vae_{latent_dim}_{0.05}_{domain_size}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# ------------------------ latent space sampling -------------------------
samples = 1000
scale = 2.0

start = scale * torch.ones((1, latent_dim), dtype=torch.float32, device=device)
end = -scale * torch.ones((1, latent_dim), dtype=torch.float32, device=device)

t = torch.linspace(0, 1, samples).unsqueeze(1).to(device)

z = start + (end - start) * t
with torch.no_grad():
    gen_shapes = standardizex.inverse(model.decode(z))
gen_shapes = gen_shapes.squeeze().cpu()

# ----------------------- animation postprocessing -----------------------
for i in range(samples):
    fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
    ax.imshow(gen_shapes[i].T, cmap="binary", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig(
        BASE_DIR
        / f"../../results/animations/animation_frames/vae_high_latent_{latent_dim}/frame_{i}.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
