import argparse
from pathlib import Path

import torch
from torchvision.io.image import decode_image
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import resnet50, ResNet50_Weights

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
IMAGES = ["rhinos.jpg", "fish.jpg", "monkeys.jpg", "turtles.jpg"]
TOP_K = 5

# --------------------------------- instantiate model ---------------------------------
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
weights = EfficientNet_V2_L_Weights.DEFAULT
model = efficientnet_v2_l(weights=weights)
model.eval().to(device)
preprocess = weights.transforms()  # crop, rescale & normalization

# ----------------------------------- classification ----------------------------------
for image in IMAGES:
    raw_img = decode_image(str(DATA_DIR / "images" / image))  # (channels, h, w)
    path = str(RESULTS_DIR / f"classification_{image}") if args.book else None
    show_image(raw_img.permute(1, 2, 0).numpy(), path=path, close=args.book)

    x = preprocess(raw_img).unsqueeze(0).to(device)  # (1, 3, s, s)

    y_pred = model(x)
    prob_pred = torch.nn.functional.softmax(y_pred, dim=1).squeeze()

    top_prob, top_class_id = torch.topk(prob_pred, TOP_K)
    for i in range(TOP_K):
        score = top_prob[i].item()
        category_name = weights.meta["categories"][top_class_id[i]]
        print(f"{i + 1}: {category_name:<20} {100 * score:.2f}%")
