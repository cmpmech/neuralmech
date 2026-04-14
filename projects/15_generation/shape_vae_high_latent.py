from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- load trained model --------------------------
domain_size = 128
latent_dim = 2

model = torch.load(
    BASE_DIR / f"../../models/shape_vae_{latent_dim}_{0.05}_{domain_size}.pt2",
    weights_only=False,
    map_location=device,
)
model.eval()
standardizex = model.standardizer

# ------------------------ latent space sampling -------------------------
samples = 16
scale = 0.5

start = scale * torch.ones((1, latent_dim), dtype=torch.float32, device=device)
end = -scale * torch.ones((1, latent_dim), dtype=torch.float32, device=device)

t = torch.linspace(0, 1, samples).unsqueeze(1).to(device)

z = start + (end - start) * t
with torch.no_grad():
    gen_shapes = standardizex.inverse(model.decode(z))
gen_shapes = gen_shapes.squeeze().cpu()

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(1, samples, figsize=(samples, 1), dpi=domain_size)
for i in range(samples):
    ax[i].imshow(gen_shapes[i].T, cmap="binary", vmin=0, vmax=1, origin="lower")
    ax[i].axis("off")
plt.show()
