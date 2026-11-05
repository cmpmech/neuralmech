import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MODEL = "google/vit-base-patch16-224"  # 16 x 16 patches of a 224 x 224 image
IMAGE = "goose.jpg"
TOP = 5

# ------------------------------------- load image ------------------------------------
image = Image.open(DATA_DIR / "images" / IMAGE).convert("RGB")
side = min(image.size)
left, top = (image.width - side) // 2, (image.height - side) // 2
image = image.crop((left, top, left + side, top + side)).resize((224, 224))

# ------------------------------------- load model ------------------------------------
processor = ViTImageProcessor.from_pretrained(MODEL)
model = ViTForImageClassification.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

# ------------------------------------- inference -------------------------------------
x = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
with torch.no_grad():
    output = model(x, output_attentions=True)
probabilities = torch.softmax(output.logits[0], dim=-1).cpu()
values, ids = probabilities.topk(TOP)
labels = [model.config.id2label[i].split(",")[0] for i in ids.tolist()]

# attention rollout: product of head-averaged attention plus identity over all layers
rollout = torch.eye(197, device=device)
for layer in output.attentions:
    a = layer[0].mean(0) + torch.eye(197, device=device)
    rollout = (a / a.sum(-1, keepdim=True)) @ rollout
attention = rollout[0, 1:].reshape(14, 14).cpu().numpy()  # class token to patch tokens
attention = np.clip(attention / np.percentile(attention, 98), 0, 1)  # few artifact tokens dominate the max

# ----------------------------------- postprocessing ----------------------------------
for label, value in zip(labels, values.tolist()):
    print(f"{label}: {value:.3f}")

overlay = np.kron(attention, np.ones((16, 16)))
if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[1].imshow(image)
    axes[1].imshow(overlay, cmap="cividis", alpha=0.6)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    image.save(RGB_PNG_DIR / "vit_input.png")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.asarray(image.convert("L")), cmap="gray")
    ax.imshow(overlay, cmap="cividis", alpha=0.7)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PNG_DIR / "vit_attention.png", dpi=150)
    save_csv(CSV_DIR / "vit_top5.csv", rank=np.arange(1, TOP + 1), label=labels, p=values.numpy())
