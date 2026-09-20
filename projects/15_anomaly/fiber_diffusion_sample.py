from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from DL import sinusoidal_embedding

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
SAMPLES = 4
RESOLUTION = 256
TIME_CHANNELS = 32
MODEL = "fiber_diffusion_200_256.pt2"

# ------------------------------------- load model ------------------------------------
model = torch.load(MODEL_DIR / MODEL, weights_only=False, map_location=device)
model.eval()
alpha_bars_prev = torch.cat([torch.ones(1, device=device), model.alpha_bars[:-1]])


def denoise(x, t):
    embedding = model["time"](sinusoidal_embedding(t, TIME_CHANNELS))
    return model["head"](model["unet"](x, embedding))


# -------------------------------------- sampling -------------------------------------
x = torch.randn(SAMPLES, 1, RESOLUTION, RESOLUTION, device=device)
with torch.no_grad():
    steps = reversed(range(model.T))
    for step in tqdm(steps, total=model.T, desc="Sampling: ", ncols=90):
        t = torch.full((SAMPLES,), step, device=device)
        eps_pred = denoise(x, t)
        a_bar = model.alpha_bars[step]
        x0_pred = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        x0_pred = x0_pred.clamp(-1, 1)
        x = (
            torch.sqrt(alpha_bars_prev[step]) * model.betas[step] * x0_pred
            + torch.sqrt(model.alphas[step]) * (1 - alpha_bars_prev[step]) * x
        ) / (1 - a_bar)
        if step > 0:
            x = x + torch.sqrt(model.posterior_variance[step]) * torch.randn_like(x)
x = ((x.clamp(-1, 1) + 1) / 2).cpu()

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots(1, SAMPLES, figsize=(2 * SAMPLES, 2))
for i in range(SAMPLES):
    ax[i].imshow(x[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[i].set_aspect("equal")
    ax[i].axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
