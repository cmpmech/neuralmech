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
IMAGE = "goose.jpg"
TARGET = "toaster"  # class of the adversarial attack

# attack, as in adversarial_perturbation.py
EPSILON = 1 / 255
STEPS = 20
STEP_SIZE = EPSILON / 5

# smoothgrad
SAMPLES = 32
NOISE = 0.1  # standard deviation of the input noise, pixels live in [0, 1]

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
def predict(x):
    with torch.no_grad():
        prob = torch.softmax(network(F.normalize(x, mean, std)), dim=1).squeeze()
    top_prob, top_id = prob.max(dim=0)
    return top_id.item(), top_prob.item()


def smoothgrad(x, class_id):
    saliency = torch.zeros_like(x[0, 0])
    for sample in range(SAMPLES):
        x_noisy = (x + NOISE * torch.randn_like(x)).requires_grad_(True)
        logit = network(F.normalize(x_noisy, mean, std))[0, class_id]
        logit.backward()
        saliency += x_noisy.grad[0].abs().amax(dim=0)
    saliency = saliency / SAMPLES
    saliency = saliency.clamp(max=torch.quantile(saliency, 0.99))  # a few pixels dominate otherwise
    return (saliency / saliency.max()).cpu().numpy()


# ------------------------------------ load image -------------------------------------
raw_img = decode_image(str(DATA_DIR / "images" / IMAGE))
x = F.center_crop(F.resize(raw_img, size, antialias=True), size)
x = (x.float() / 255).unsqueeze(0).to(device)

# targeted projected gradient descent, see adversarial_perturbation.py
target_id = torch.tensor([categories.index(TARGET)], device=device)
delta = torch.zeros_like(x, requires_grad=True)
for step in range(STEPS):
    cost = nn.functional.cross_entropy(network(F.normalize(x + delta, mean, std)), target_id)
    cost.backward()
    with torch.no_grad():
        delta -= STEP_SIZE * delta.grad.sign()
        delta.clamp_(-EPSILON, EPSILON)
        delta.copy_((x + delta).clamp(0, 1) - x)
    delta.grad.zero_()
x_adv = (x + delta).detach()

# ------------------------------------ explanation ------------------------------------
saliencies = {}
for name, image in {"original": x, "attacked": x_adv}.items():
    class_id, prob = predict(image)
    saliencies[name] = smoothgrad(image, class_id)
    print(f"{name}: explains {categories[class_id]} {100 * prob:.2f}%")

# ----------------------------------- postprocessing ----------------------------------
original = x.squeeze().permute(1, 2, 0).cpu().numpy()
cmap = plt.get_cmap("cividis")

if not args.book:
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    axs[0].imshow(original)
    for ax, (name, saliency) in zip(axs[1:], saliencies.items()):
        ax.imshow(saliency, cmap=cmap)
        ax.set_title(name)
    for ax in axs:
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.02)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    show_image(original, path=RGB_PDF_DIR / "saliency_image.pdf", close=True)
    for name, saliency in saliencies.items():
        show_image(cmap(saliency)[..., :3], path=RGB_PDF_DIR / f"saliency_{name}.pdf", close=True)
