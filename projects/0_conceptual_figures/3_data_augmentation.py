import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# ------------------------------ load image ------------------------------
img = Image.open(DATA_DIR / "images/pasta.jpg").convert("RGB")

# ------------------------------- original -------------------------------
show_image(np.asarray(img), path=RESULTS_DIR / "augment_0.png", close=args.book)

# ---------------------------- augmentations -----------------------------
augment1 = T.RandomHorizontalFlip(p=1)
show_image(
    np.asarray(augment1(img)), path=RESULTS_DIR / "augment_1.png", close=args.book
)
augment2 = T.ColorJitter(brightness=0.5, contrast=1.0, saturation=0.5, hue=0.2)
show_image(
    np.asarray(augment2(img)), path=RESULTS_DIR / "augment_2.png", close=args.book
)
augment3 = T.CenterCrop(size=(img.size[1], img.size[1]))
show_image(
    np.asarray(augment3(img)), path=RESULTS_DIR / "augment_3.png", close=args.book
)
augment4 = T.Compose(
    [T.RandomRotation(degrees=(90, 90)), T.CenterCrop((img.size[1], img.size[1]))]
)
show_image(
    np.asarray(augment4(img)), path=RESULTS_DIR / "augment_4.png", close=args.book
)
augment5 = T.GaussianBlur(kernel_size=15, sigma=15.0)
show_image(
    np.asarray(augment5(img)), path=RESULTS_DIR / "augment_5.png", close=args.book
)


def augment6(img):
    img = T.ToTensor()(img)
    noise = torch.randn_like(img) * 0.2
    img = torch.clamp(img + noise, 0, 1)
    return T.ToPILImage()(img)


show_image(
    np.asarray(augment6(img)), path=RESULTS_DIR / "augment_6.png", close=args.book
)


def augment7(img, strength=0.2, radius=0.7, ripples=20):
    img_t = T.ToTensor()(img).unsqueeze(0)
    _, _, H, W = img_t.shape
    x0, y0 = -0.5, 0.5

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )

    dx0 = xx - x0
    dy0 = yy - y0
    r = torch.sqrt(dx0**2 + dy0**2) + 1e-6

    mask = torch.clamp(1 - r / radius, 0, 1)
    disp = strength * mask * torch.sin(ripples * torch.pi * r)
    grid = torch.stack((xx + disp * dx0 / r, yy + disp * dy0 / r), dim=-1).unsqueeze(0)

    out = F.grid_sample(img_t, grid, align_corners=True).squeeze(0)
    return T.ToPILImage()(out)


show_image(
    np.asarray(augment7(img)), path=RESULTS_DIR / "augment_7.png", close=args.book
)
