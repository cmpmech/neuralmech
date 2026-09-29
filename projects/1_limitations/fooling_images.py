import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.transforms.v2 import functional as F

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# fooling
TARGETS = ["goose", "goldfish", "toaster", "tabby"]  # classes the noise is pushed toward
STEPS = 50
STEP_SIZE = 2 / 255
EPSILON = 8 / 255  # largest distance from the starting noise

# --------------------------------- instantiate model ---------------------------------
weights = EfficientNet_V2_L_Weights.DEFAULT
categories = weights.meta["categories"]
size = weights.transforms().crop_size[0]
mean = weights.transforms().mean
std = weights.transforms().std

network = efficientnet_v2_l(weights=weights)
network.eval().to(device)
network.requires_grad_(False)


# -------------------------------------- helper ---------------------------------------
def model(x):
    return network(F.normalize(x, mean, std))


def predict(x):
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1).squeeze()
    top_prob, top_id = prob.max(dim=0)
    return categories[top_id], top_prob.item()


# ------------------------------------ create data ------------------------------------
noise = torch.rand(1, 3, size, size, device=device)
images = {"noise": noise}

# gradient ascent on the target class, starting from the noise
for target in TARGETS:
    target_id = torch.tensor([categories.index(target)], device=device)
    delta = torch.zeros_like(noise, requires_grad=True)
    for step in range(STEPS):
        cost = nn.functional.cross_entropy(model(noise + delta), target_id)
        cost.backward()
        with torch.no_grad():
            delta -= STEP_SIZE * delta.grad.sign()
            delta.clamp_(-EPSILON, EPSILON)
            delta.copy_((noise + delta).clamp(0, 1) - noise)
        delta.grad.zero_()
    images[f"noise_{target}"] = (noise + delta).detach()

predictions = {name: predict(image) for name, image in images.items()}
for name, (label, prob) in predictions.items():
    print(f"{name}: {label} {100 * prob:.2f}%")

# ----------------------------------- postprocessing ----------------------------------
panels = {name: image.squeeze().permute(1, 2, 0).cpu().numpy() for name, image in images.items()}

if not args.book:
    fig, axs = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for ax, (name, panel) in zip(axs, panels.items()):
        label, prob = predictions[name]
        ax.imshow(panel)
        ax.set_title(f"{label}\n{100 * prob:.1f}%", fontsize=9)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0, wspace=0.05)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, panel in panels.items():
        show_image(panel, path=RGB_PDF_DIR / f"fooling_{name}.pdf", close=True)
