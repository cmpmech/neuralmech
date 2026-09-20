from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
SAMPLES = 4
PRIOR_LATENT = 64
THRESHOLD = 0.5
MODEL = "fiber_vae_256_4.0_256.pt2"
TEMPERATURE = 0.5  # cheating

# ------------------------------------- load model ------------------------------------
model = torch.load(MODEL_DIR / MODEL, weights_only=False, map_location=device)
model.eval()

# -------------------------------------- sampling -------------------------------------
with torch.no_grad():
    z = model.prior.decode(
        torch.randn(SAMPLES, PRIOR_LATENT, device=device) * TEMPERATURE
    )
    x = model.decode(model.standardizez.inverse(z))
x = (model.standardizer.inverse(x) >= THRESHOLD).cpu()

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(1, SAMPLES, figsize=(2 * SAMPLES, 2))
for i in range(SAMPLES):
    ax[i].imshow(x[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[i].set_aspect("equal")
    ax[i].axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
