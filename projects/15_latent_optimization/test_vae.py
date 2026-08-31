from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cpu")  # a single decoder pass
torch.manual_seed(1)  # change to draw a different sample
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# draw one code from the prior of the variational autoencoder and decode it. a prior
# sample only decodes into a realistic microstructure if the kl term actually shapes the
# latent, which is what this checks
RESOLUTION = 256
LATENT = 64  # matches ../15_anomaly/fiber_vae_train.py
# LATENT = 8**2 * 4 + 4  # matches ../15_anomaly/fiber_vae_train_v2.py
THRESHOLD = 0.5

# --------------------------------- instantiate model ---------------------------------
model = torch.load(
    MODEL_DIR / "fiber_vae_64_1.0_256.pt2", weights_only=False, map_location=device
)
# the slot-latent variant, swap both this and LATENT above to test it
# model = torch.load(
#     MODEL_DIR / "fiber_vae_v2_8_4_32.0_256.pt2", weights_only=False, map_location=device
# )
model.eval()

# --------------------------------------- sample --------------------------------------
latent = torch.randn(1, LATENT)
with torch.no_grad():
    fibers = torch.sigmoid(model.decode(latent.to(device)))  # decoder emits bce logits

print(f"latent distance {torch.linalg.norm(latent):.1f}")
print(f"fiber area fraction {(fibers >= THRESHOLD).to(fibers.dtype).mean():.3f}")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=RESOLUTION)
ax.imshow(fibers[0, 0], cmap="binary", vmin=0, vmax=1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
