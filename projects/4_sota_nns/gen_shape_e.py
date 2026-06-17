import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR.parent.parent / "results" / "shape_e"
WEIGHTS_DIR = BASE_DIR.parent.parent / "external_data" / "shape_e_weights"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# route Hugging Face downloads into external_data (must precede diffusers import)
os.environ["HF_HOME"] = str(WEIGHTS_DIR)

import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# ------------------------------- settings -------------------------------
MODEL_ID = "openai/shap-e"
PROMPTS = [
    "a wooden chair",
    "a ceramic vase",
    "a sports car",
    "a potted plant",
    "a coffee mug",
]
N_SHAPES = len(PROMPTS)
GUIDANCE_SCALE = 15.0
N_INFERENCE_STEPS = 64
FRAME_SIZE = 256

# fp16 is GPU-only; fall back to fp32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32

# ------------------------------- pipeline -------------------------------
pipe = ShapEPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
pipe = pipe.to(device)

# -------------------------- generate & export ---------------------------
for i in range(N_SHAPES):
    prompt = PROMPTS[i]
    generator = torch.Generator(device=device).manual_seed(i)

    mesh = pipe(
        prompt,
        generator=generator,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=N_INFERENCE_STEPS,
        frame_size=FRAME_SIZE,
        output_type="mesh",
    ).images[0]

    out_path = RESULTS_DIR / f"shape_{i}.ply"
    export_to_ply(mesh, str(out_path))
    print(f'"{prompt}" -> {out_path}')
