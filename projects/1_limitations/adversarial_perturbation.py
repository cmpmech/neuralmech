import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.io.image import decode_image
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.transforms.v2 import functional as F

from postprocessing import show_image

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
IMAGES = {"goose.jpg": "toaster", "fish.jpg": "banana"}
EPSILON = 1 / 255  # largest change per pixel and channel, images live in [0, 1]
STEPS = 20
STEP_SIZE = EPSILON / 5
AMPLIFICATION = 50  # makes the perturbation visible

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
    return network(F.normalize(x, mean, std))  # attack in pixel space, not normalized space


def predict(x):
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1).squeeze()
    top_prob, top_id = prob.max(dim=0)
    return categories[top_id], top_prob.item()


# --------------------------------------- attack --------------------------------------
for image, target in IMAGES.items():
    raw_img = decode_image(str(DATA_DIR / "images" / image))
    x = F.center_crop(F.resize(raw_img, size, antialias=True), size)
    x = (x.float() / 255).unsqueeze(0).to(device)  # (1, 3, s, s)
    target_id = torch.tensor([categories.index(target)], device=device)

    # targeted projected gradient descent in the infinity-norm ball
    delta = torch.zeros_like(x, requires_grad=True)
    for step in range(STEPS):
        cost = nn.functional.cross_entropy(model(x + delta), target_id)
        cost.backward()
        with torch.no_grad():
            delta -= STEP_SIZE * delta.grad.sign()
            delta.clamp_(-EPSILON, EPSILON)
            delta.copy_((x + delta).clamp(0, 1) - x)
        delta.grad.zero_()
    x_adv = (x + delta).detach()

    label, prob = predict(x)
    label_adv, prob_adv = predict(x_adv)
    print(f"{image}: {label} {100 * prob:.2f}% -> {label_adv} {100 * prob_adv:.2f}%")

# ----------------------------------- postprocessing ----------------------------------
    stem = Path(image).stem
    original = x.squeeze().permute(1, 2, 0).cpu().numpy()
    adversarial = x_adv.squeeze().permute(1, 2, 0).cpu().numpy()
    perturbation = (0.5 + AMPLIFICATION * delta.detach()).clamp(0, 1)
    perturbation = perturbation.squeeze().permute(1, 2, 0).cpu().numpy()

    if not args.book:
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.3))
        panels = [original, perturbation, adversarial]
        titles = [f"{label} {100 * prob:.1f}%", f"perturbation x{AMPLIFICATION}",
                  f"{label_adv} {100 * prob_adv:.1f}%"]
        for ax, panel, title in zip(axs, panels, titles):
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.02)
        plt.show()
# -------------------------------- book postprocessing --------------------------------
    else:
        show_image(original, path=RGB_PDF_DIR / f"adversarial_{stem}.pdf", close=True)
        show_image(perturbation, path=RGB_PDF_DIR / f"adversarial_{stem}_delta.pdf", close=True)
        show_image(adversarial, path=RGB_PDF_DIR / f"adversarial_{stem}_attacked.pdf", close=True)

