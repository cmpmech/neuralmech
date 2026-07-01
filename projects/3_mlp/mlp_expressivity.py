import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from DL import flatten_params, get_params, set_params, unflatten_params
from NN import MLP
from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = (RESULTS_DIR / "animations/animation_frames").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --------------------------------------- helper --------------------------------------
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-10, b=10)
        torch.nn.init.uniform_(m.bias, a=-10, b=10)


class SinActivation(torch.nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# -------------------------------------- settings -------------------------------------
# depth study
HIDDEN_LAYERS = 8  # 1, 2, 4, 8
NEURONS = 128

# width study
# HIDDEN_LAYERS = 2
# NEURONS = 337  # 30, 128, 221, 337

# more samples to resolve the finer features of the deeper networks
SAMPLES = {1: 200, 2: 400, 4: 800, 8: 1400}.get(HIDDEN_LAYERS, 400)

# model settings
ACTIVATION, ACTIVATION_ID = torch.nn.Sigmoid(), 0
# ACTIVATION, ACTIVATION_ID = torch.nn.Tanh(), 1
# ACTIVATION, ACTIVATION_ID = torch.nn.Mish(), 2
# ACTIVATION, ACTIVATION_ID = SinActivation(0.05), 3

LAYERS = [2] + [NEURONS] * HIDDEN_LAYERS + [3]
ACTIVATIONS = [ACTIVATION for _ in range(HIDDEN_LAYERS)]

parameters = (
    NEURONS * 8 + NEURONS * NEURONS * (HIDDEN_LAYERS - 1) + NEURONS * HIDDEN_LAYERS
)
print(f"parameters: {parameters}")

# ------------------------------------ prepare data -----------------------------------
x = torch.linspace(-1, 1, SAMPLES)
y = torch.linspace(-1, 1, SAMPLES)
x, y = torch.meshgrid(x, y, indexing="ij")
mlp_input = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)

# --------------------------------- instantiate model ---------------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS)
model.to(device)
model.apply(init_weights)


# ---------------------------------- model prediction ---------------------------------
def predict_rgb(mlp_input, norm=None):
    with torch.no_grad():
        z_pred = model(mlp_input).reshape(SAMPLES, SAMPLES, 3).cpu()
    if norm is None:
        norm = torch.stack([z_pred.amin(dim=(0, 1)), z_pred.amax(dim=(0, 1))])
    z_pred = (z_pred - norm[0]) / (norm[1] - norm[0]).clamp(min=1e-8)
    return z_pred, norm


z_pred, norm1 = predict_rgb(mlp_input)

if not args.animate:
    if not args.book:
# ----------------------------------- postprocessing ----------------------------------
        show_image(z_pred.numpy())
# -------------------------------- book postprocessing --------------------------------
    else:
        path = RGB_PDF_DIR / f"expressivity_{ACTIVATION_ID}_{HIDDEN_LAYERS}_{NEURONS}.pdf"
        show_image(z_pred.numpy(), path=path, close=True)

# ---------------------------- interpolation for animation ----------------------------
if args.animate:
    ALPHA_RANGE = 1.0
    FRAMES = 1000

    params_shape = get_params(model)
    params1 = flatten_params(params_shape)
    model.apply(init_weights)
    params2 = flatten_params(get_params(model))
    z_pred, norm2 = predict_rgb(mlp_input)

    norm = torch.stack(
        [
            torch.min(norm1[0], norm2[0]),  # per-channel mins
            torch.max(norm1[1], norm2[1]),  # per-channel maxes
        ]
    )

    folder = f"expressivity_{ACTIVATION_ID}_{HIDDEN_LAYERS}_{NEURONS}"
    (ANIMATION_DIR / folder).mkdir(parents=True, exist_ok=True)

    alpha = torch.linspace(0, ALPHA_RANGE, FRAMES)
    for i, a in tqdm(enumerate(alpha), total=len(alpha), desc="frames"):
        params = params1 + a * (params2 - params1)
        set_params(model, unflatten_params(params, params_shape))

        # interpolated prediction with the combined normalization
        z_pred, _ = predict_rgb(mlp_input, norm)
        z_pred = z_pred.clamp(0, 1)  # interpolated frames sometimes exceed [0, 1]
        show_image(
            z_pred.numpy(), path=ANIMATION_DIR / f"{folder}/frame_{i}.jpg", close=True
        )

# stylization options
# in predict_rgb
# if norm is None:
#     low = z_pred.quantile(0.05, dim=0).amin(dim=0)  # (3,)
#     high = z_pred.quantile(0.95, dim=0).amax(dim=0)  # (3,)
#     norm = torch.stack([low, high])
#
# after predict_rgb
# z_pred = z_pred.clamp(0, 1).pow(1.2)
# or
# z_pred = z_pred.clamp(0, 1).pow(0.5)
#
# saturation
# z_pred = 0.5 + (z_pred - 0.5) * 1.2
#
# hue shifts (here to red)
# z_pred[:, :, 1] *= 0.6
# z_pred[:, :, 2] *= 0.6
