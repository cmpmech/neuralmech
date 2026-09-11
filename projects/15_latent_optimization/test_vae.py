from pathlib import Path

import matplotlib.pyplot as plt
import torch

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cpu")  # a single decoder pass
torch.manual_seed(1)  # change to draw a different sample
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# decode a code drawn from a standard normal next to one drawn from the prior fitted to
# the codes. the decoder is the same in both, so whatever differs between the two images
# is the prior alone, and the standard normal is the one that decodes into merged worms
RESOLUTION = 256
# LATENT = 8**2 * 4 + 4  # matches ../15_anomaly/fiber_vae_train.py
LATENT = 4**2 * 8 + 4  # matches ../15_anomaly/fiber_vae_train_v2.py
THRESHOLD = 0.5

# --------------------------------- instantiate model ---------------------------------
# model = torch.load(
#     MODEL_DIR / "fiber_vae_v2_132_1.0_256.pt2",
#     weights_only=False,
#     map_location=device,
# <<<<<<< HEAD
#     MODEL_DIR / "fiber_vae_260_1.0_256.pt2", weights_only=False, map_location=device
# =======
# <<<<<<< HEAD
#     MODEL_DIR / "fiber_vae_64_1.0_256.pt2", weights_only=False, map_location=device
# =======
#     MODEL_DIR / "fiber_vae_64_12.0_256.pt2", weights_only=False, map_location=device
# >>>>>>> eb9b070 (local minimum)
# >>>>>>> afb0bd9 (local minimum)
# )
# the variant whose prior is a second variational autoencoder rather than an
# autoregressive one, swap both this and LATENT above to compare the two
model = torch.load(
    MODEL_DIR / "fiber_vae_v2_132_1.0_256.pt2", weights_only=False, map_location=device
)
model.eval()

# --------------------------------------- sample --------------------------------------
with torch.no_grad():
    normal = torch.sigmoid(model.decode(torch.randn(1, LATENT, device=device)))
    learned = torch.sigmoid(model.decode(model.prior.sample(1)))  # decoder emits logits

for name, fibers in [("normal", normal), ("learned", learned)]:
    area = (fibers >= THRESHOLD).to(fibers.dtype).mean()
    print(f"{name} prior, fiber area fraction {area:.3f} (the data has 0.13)")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=RESOLUTION // 2)
ax[0].imshow(normal[0, 0], cmap="binary", vmin=0, vmax=1)
ax[1].imshow(learned[0, 0], cmap="binary", vmin=0, vmax=1)
for axis in ax:
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
