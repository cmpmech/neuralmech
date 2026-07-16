import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP, SIRENsine
from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 1500
LR = 2e-4

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
HIDDEN_LAYERS = 4
WIDTHS = [8, 16, 32]  # only widths whose siren beats the byte-matched jpeg at this resolution
OMEGA_0 = 30.0

# compression settings
RESOLUTION = 1200  # downsampled width in pixels; height derived to keep the source aspect ratio


# --------------------------------------- helper ---------------------------------------
def to_image(y):
    return y.clamp(0, 1).cpu().numpy()


def jpeg_at_bytes(img, target_bytes):
    best_quality, best_bytes, best_buf = None, None, None
    for quality in range(1, 96):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        size = buf.tell()
        if best_bytes is None or abs(size - target_bytes) < abs(best_bytes - target_bytes):
            best_quality, best_bytes, best_buf = quality, size, buf
    reconstructed = np.asarray(Image.open(best_buf).convert("RGB"))
    return best_quality, best_bytes, reconstructed


# ------------------------------------ load image -------------------------------------
img = Image.open(DATA_DIR / "images/fish.jpg").convert("RGB")
native_width, native_height = img.size
height = round(RESOLUTION * native_height / native_width)
img = img.resize((RESOLUTION, height))

target_uint8 = np.array(img)
target = torch.from_numpy(target_uint8).to(torch.float32).div(255.0).to(device)
raw_bytes = target_uint8.size

x = torch.linspace(-1, 1, RESOLUTION)
y = torch.linspace(-1, 1, height)
y, x = torch.meshgrid(y, x, indexing="ij")
mlp_input = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)

print(f"downsampled to {RESOLUTION}x{height}, raw {raw_bytes} bytes")

# -------------------------------------- training -------------------------------------
siren_images, siren_bytes, siren_ratios = {}, {}, {}

tic = time.time()
for width in WIDTHS:
    layers = [2] + [width] * HIDDEN_LAYERS + [3]
    activation = SIRENsine(omega_0=OMEGA_0)
    activations = [activation for _ in range(len(layers) - 2)] + [None]
    model = MLP(layers, post_modules=activations)
    model.to(device)
    init_weights(model, activation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    pbar = tqdm(range(EPOCHS), desc=f"width {width}")
    model.train()
    for epoch in pbar:
        optimizer.zero_grad()
        y_pred = model(mlp_input).reshape(height, RESOLUTION, 3)
        cost = cost_fun(y_pred, target)
        cost.backward()
        optimizer.step()
        if epoch % 100 == 0:
            pbar.set_postfix({"train": f"{cost.item():.2e}"})

    with torch.no_grad():
        y_pred = model(mlp_input).reshape(height, RESOLUTION, 3)
    siren_images[width] = to_image(y_pred)

    num_params = sum(p.numel() for p in model.parameters())
    siren_bytes[width] = num_params * 4
    siren_ratios[width] = raw_bytes / siren_bytes[width]
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
close = args.book
for width in WIDTHS:
    quality, jpeg_size, jpeg_uint8 = jpeg_at_bytes(target_uint8, siren_bytes[width])
    jpeg_ratio = raw_bytes / jpeg_size
    jpeg_image = jpeg_uint8.astype(np.float32) / 255.0

    print(
        f"width {width}: siren {siren_bytes[width]} bytes (ratio {siren_ratios[width]:.1f}x), "
        f"jpeg quality {quality} {jpeg_size} bytes (ratio {jpeg_ratio:.1f}x)"
    )

    siren_path = RGB_PDF_DIR / f"siren_image_compression_siren_w{width}.pdf" if args.book else None
    jpeg_path = RGB_PDF_DIR / f"siren_image_compression_jpeg_w{width}.pdf" if args.book else None
    show_image(siren_images[width], path=siren_path, close=close)
    show_image(jpeg_image, path=jpeg_path, close=close)

target_path = RGB_PDF_DIR / "siren_image_compression_target.pdf" if args.book else None
show_image(target_uint8.astype(np.float32) / 255.0, path=target_path, close=close)
