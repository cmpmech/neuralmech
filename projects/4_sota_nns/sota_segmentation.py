import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io.image import decode_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true", help="save figures to results/")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
IMAGE = "goose.jpg"
# IMAGE = "courtyard.jpg"
COLORS = ["Oranges_r", "Blues_r", "Reds_r", "Purples_r", "Greens_r"]

# --------------------------------- instantiate model ---------------------------------
# weights = DeepLabV3_ResNet50_Weights.DEFAULT
# model = deeplabv3_resnet50(weights=weights)
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval().to(device)
preprocess = weights.transforms()  # center crop, rescale & normalization

# ------------------------------------- load image ------------------------------------
raw_img = decode_image(str(DATA_DIR / "images" / IMAGE))  # (channels, h, w)
img_max, img_min = raw_img.max(), raw_img.min()
path = str(RESULTS_DIR / "segmentation1.jpg") if args.book else None
show_image(raw_img.permute(1, 2, 0).numpy(), path=path, close=args.book)

x = preprocess(raw_img).unsqueeze(0).to(device)

# --------------------------------------- predict -------------------------------------
y_pred = model(x)["out"]  # 'aux' head not needed during inference

normalized_masks = y_pred.softmax(dim=1)
class_map = normalized_masks.argmax(dim=1).squeeze().cpu()
unique_class_ids = torch.unique(class_map)

# ----------------------------------- postprocessing ----------------------------------
# rescale the normalized network input back to the raw 8-bit intensity range
input_img = (x[0].cpu() - x.min()) / (x.max() - x.min())
input_img = (input_img * (img_max - img_min) + img_min).to(torch.uint8)
input_img = input_img.permute(1, 2, 0).numpy()

h, w = input_img.shape[:2]
fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
ax.imshow(input_img)
for i, class_id in enumerate(unique_class_ids):
    mask = class_map == class_id
    name = weights.meta["categories"][class_id]
    print(f'class "{name}" covers {mask.sum()} pixels')

    overlay = np.ma.masked_where(~mask, mask)
    ax.imshow(overlay, alpha=0.5, cmap=COLORS[i % len(COLORS)])
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

if args.book:
    plt.savefig(RESULTS_DIR / "segmentation2.jpg")
else:
    plt.show()
