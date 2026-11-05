import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = ("This image shows a result of a 2D simulation. What could the colors represent, "
          "and what do the boundary conditions and the geometry look like? Answer in three sentences.")
MAX_TOKENS = 200

# ------------------------------------- load image ------------------------------------
field = np.flipud(np.load(DATA_DIR / "elasticity2D_reference.npz")["uy_1"])
field = (field - np.nanmin(field)) / (np.nanmax(field) - np.nanmin(field))
rgb = colormaps["turbo"](np.nan_to_num(field))[..., :3]
rgb[np.isnan(field)] = 1.0  # hole in white
image = Image.fromarray((rgb * 255).astype(np.uint8))

# ------------------------------------- load model ------------------------------------
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()

# ------------------------------------- inference -------------------------------------
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
answer = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

# ----------------------------------- postprocessing ----------------------------------
print(f"image tokens: {(inputs['input_ids'] == model.config.image_token_id).sum().item()}")
print(answer)

if not args.book:
    plt.imshow(image)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    image.save(RGB_PNG_DIR / "vlm_input.png")
    (RESULTS_DIR / "data" / "vlm_description.txt").write_text(f"prompt: {PROMPT}\nanswer: {answer}\n")
