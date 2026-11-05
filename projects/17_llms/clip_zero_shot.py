import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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
MODEL = "openai/clip-vit-base-patch32"
PROMPTS = [
    "a photo of a bird",
    "a photo of a building",
    "a photo of food",
    "a finite element simulation of a deformed plate",
    "a computed tomography scan",
    "a fiber-reinforced composite microstructure",
]


# --------------------------------------- helper --------------------------------------
def square(image, size=224):
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size))


def render(field, cmap):
    field = (field - np.nanmin(field)) / (np.nanmax(field) - np.nanmin(field))
    rgb = colormaps[cmap](np.nan_to_num(field))[..., :3]
    rgb[np.isnan(field)] = 1.0  # holes in white
    return square(Image.fromarray((rgb * 255).astype(np.uint8)))


# ------------------------------------- load data -------------------------------------
images = [square(Image.open(DATA_DIR / "images" / name).convert("RGB"))
          for name in ["goose.jpg", "kanazawa_castle.jpg", "pasta.jpg"]]
plate = np.load(DATA_DIR / "elasticity2D_reference.npz")["uy_1"]
images.append(render(np.flipud(plate), "turbo"))
images.append(render(np.load(DATA_DIR / "ct_slice.npy")[240:720], "gray"))
images.append(render(np.load(DATA_DIR / "fibers_256.npy")[0], "binary"))

# ------------------------------------- load model ------------------------------------
processor = CLIPProcessor.from_pretrained(MODEL)
model = CLIPModel.from_pretrained(MODEL).to(device)
model.eval()

# ------------------------------------- inference -------------------------------------
inputs = processor(text=PROMPTS, images=images, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    z_image = model.get_image_features(pixel_values=inputs["pixel_values"])
    z_text = model.get_text_features(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"])
z_image = z_image / z_image.norm(dim=-1, keepdim=True)
z_text = z_text / z_text.norm(dim=-1, keepdim=True)
similarity = (z_image @ z_text.T).cpu()  # cosine similarity, images x prompts
probabilities = torch.softmax(model.logit_scale.exp().item() * similarity, dim=-1).numpy()

# ----------------------------------- postprocessing ----------------------------------
for i, row in enumerate(probabilities):
    print(f"image {i}: {PROMPTS[row.argmax()]} ({row.max():.2f})")

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(np.concatenate([np.asarray(im) for im in images], axis=1))
    axes[1].imshow(probabilities, cmap="cividis")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for i, image in enumerate(images):
        image.save(RGB_PNG_DIR / f"clip_image_{i}.png")
    i, j = np.meshgrid(np.arange(len(images)), np.arange(len(PROMPTS)), indexing="ij")
    save_csv(CSV_DIR / "clip_zero_shot.csv", image=i.flatten() + 1, prompt=j.flatten() + 1,
             p=probabilities.flatten(), s=similarity.numpy().flatten())
