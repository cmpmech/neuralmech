import argparse
from pathlib import Path

import torch
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

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
IMAGE = "gooses.jpg"
# IMAGE = "firebrigade.jpg"
SCORE_THRESHOLD = 0.8
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --------------------------------- instantiate model ---------------------------------
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_THRESHOLD)
model.eval().to(device)
preprocess = weights.transforms()  # center crop, rescale & normalization

# ------------------------------------- load image ------------------------------------
raw_img = decode_image(str(DATA_DIR / "images" / IMAGE))  # (channels, h, w)
img_max, img_min = raw_img.max(), raw_img.min()
path = str(RESULTS_DIR / "objectdetection1.jpg") if args.book else None
show_image(raw_img.permute(1, 2, 0).numpy(), path=path, close=args.book)

x = preprocess(raw_img).unsqueeze(0).to(device)

# --------------------------------------- predict -------------------------------------
y_pred = model(x)[0]  # one dict per image in the batch

keep = y_pred["scores"] > SCORE_THRESHOLD
boxes = y_pred["boxes"][keep]
labels = y_pred["labels"][keep]
scores = y_pred["scores"][keep]

pred_labels = []
for i in range(len(boxes)):
    score = scores[i].item()
    label_name = weights.meta["categories"][labels[i].item()]
    pred_labels.append(f"{label_name}: {100 * score:.1f}%")
    print(f"{i + 1}: {label_name:<20} {100 * score:.1f}%")

# ----------------------------------- postprocessing ----------------------------------
# rescale the normalized network input back to the raw 8-bit intensity range
norm_img = x[0].cpu()
img_uint8 = ((norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
             * (img_max - img_min) + img_min).to(torch.uint8)

box_img = draw_bounding_boxes(
    img_uint8,
    boxes=boxes,
    labels=pred_labels,
    colors="white",
    font=FONT,
    width=4,
    font_size=44,
)

path = str(RESULTS_DIR / "objectdetection2.jpg") if args.book else None
show_image(box_img.permute(1, 2, 0).numpy(), path=path, close=args.book)
