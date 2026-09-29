import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io.image import decode_image
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.transforms.v2 import functional as F

from postprocessing import save_csv, show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
IMAGES = ["duckling.jpg"]  # also tried: goose, deer, beijing_facade, kanazawa_castle, firebrigade, su
ANGLES = np.arange(0, 361, 5)
SNAPSHOTS = [0, 45, 90, 135, 180, 225, 270, 315]  # angles shown as images

# --------------------------------- instantiate model ---------------------------------
weights = EfficientNet_V2_L_Weights.DEFAULT
categories = weights.meta["categories"]
preprocess = weights.transforms()
model = efficientnet_v2_l(weights=weights)
model.eval().to(device)

# ----------------------------------- classification ----------------------------------
for image in IMAGES:
    raw_img = decode_image(str(DATA_DIR / "images" / image))
    size = min(raw_img.shape[1:])
    raw_img = F.center_crop(raw_img, size)  # square, so rotation keeps the whole object

    # a circular mask removes the corners that rotation would otherwise reveal
    grid = torch.linspace(-1, 1, size)
    mask = (grid[:, None] ** 2 + grid[None, :] ** 2 <= 1).to(raw_img.dtype)

    true_id = None
    true_prob = np.zeros(len(ANGLES))
    top_labels = []
    top_probs = []
    for i, angle in enumerate(ANGLES):
        rotated = F.rotate(raw_img, float(angle), interpolation=F.InterpolationMode.BILINEAR)
        rotated = rotated * mask + 255 * (1 - mask)
        with torch.no_grad():
            prob = torch.softmax(model(preprocess(rotated).unsqueeze(0).to(device)), dim=1)
        prob = prob.squeeze().cpu()
        true_id = prob.argmax().item() if true_id is None else true_id
        true_prob[i] = prob[true_id].item()
        top_labels.append(categories[prob.argmax()])
        top_probs.append(prob.max().item())
        if true_prob[i] == true_prob[: i + 1].min():
            worst = rotated
        if angle in SNAPSHOTS and args.book:
            stem = Path(image).stem
            show_image(rotated.permute(1, 2, 0).numpy(),
                       path=RGB_PDF_DIR / f"rotation_{stem}_{angle}.pdf", close=True)
    print(image, categories[true_id])
    for angle, label, top_prob, p in zip(ANGLES, top_labels, top_probs, true_prob):
        if angle % 45 == 0:
            print(f"  {angle:3d}: {label} {100 * top_prob:.1f}% (reference {100 * p:.1f}%)")

# ----------------------------------- postprocessing ----------------------------------
    if not args.book:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3), width_ratios=[2, 1])
        axs[0].plot(ANGLES, true_prob, "k")
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("rotation angle")
        axs[0].set_ylabel(f"p({categories[true_id]})")
        axs[1].imshow(worst.permute(1, 2, 0).numpy())
        axs[1].axis("off")
        fig.subplots_adjust(left=0.08, right=1, top=0.95, bottom=0.18)
        plt.show()
# -------------------------------- book postprocessing --------------------------------
    else:
        save_csv(CSV_DIR / f"rotation_{Path(image).stem}.csv", angle=ANGLES, prob=true_prob)
