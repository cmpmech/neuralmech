import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from DL import flatten_params, get_params, set_params, unflatten_params
from NN import MLP
from postprocessing import show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
activation_id = None


# -------------------------------- helper --------------------------------
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-10, b=10)
        torch.nn.init.uniform_(m.bias, a=-10, b=10)


class SinActivation(torch.nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)  # 0.1


# --------------------------- hyperparameters ----------------------------
# depth study
HIDDEN_LAYERS = 8  # 2  # 1, 2, 4, 8
NEURONS = 128

# width study
# HIDDEN_LAYERS = 2
# NEURONS = 337  # 30, 128, 221, 337

parameters = (
    NEURONS * 8 + NEURONS * NEURONS * (HIDDEN_LAYERS - 1) + NEURONS * HIDDEN_LAYERS
)
print(f"parameters: {parameters}")

samples = 400
if HIDDEN_LAYERS == 1:
    samples = 200
elif HIDDEN_LAYERS == 2:
    samples = 400
elif HIDDEN_LAYERS == 4:
    samples = 800
elif HIDDEN_LAYERS == 8:
    samples = 1400

ACTIVATION, activation_id = torch.nn.Sigmoid(), 0
# ACTIVATION, activation_id = torch.nn.Tanh(), 1
# ACTIVATION, activation_id = torch.nn.Mish(), 2
# ACTIVATION, activation_id = SinActivation(0.05), 3

# ---------------------------- preprocessing -----------------------------
layers = [2] + [NEURONS] * HIDDEN_LAYERS + [3]
activations = [ACTIVATION] * HIDDEN_LAYERS

x = torch.linspace(-1, 1, samples)
y = torch.linspace(-1, 1, samples)
x, y = torch.meshgrid(x, y, indexing="ij")
mlp_input = torch.cat((x.flatten().unsqueeze(1), y.flatten().unsqueeze(1)), 1).to(
    device
)

# --------------------------- model prediction ---------------------------
model = MLP(layers, activations)
model.apply(init_weights)
model.to(device)


def predict_rgb(mlp_input, norm=None):
    with torch.no_grad():
        z_pred = model(mlp_input).reshape(samples, samples, 3).cpu()
    if norm is None:
        norm = torch.stack([z_pred.amin(dim=(0, 1)), z_pred.amax(dim=(0, 1))])
    z_pred = (z_pred - norm[0]) / (norm[1] - norm[0]).clamp(min=1e-8)

    return z_pred, norm


z_pred, norm1 = predict_rgb(mlp_input)


if not args.animate:
    if args.book:
# ------------------------- book postprocessing --------------------------
        path = (
            RESULTS_DIR / f"expressivity_{activation_id}_{HIDDEN_LAYERS}_{NEURONS}.png"
        )
        close = True
    else:
# ---------------------------- postprocessing ----------------------------
        path = None
        close = False

    show_image(z_pred.numpy(), path=path, close=close)

# --------------------- interpolation for animation ----------------------
if args.animate:
    ALPHA_RANGE = 1.0  # interpolation
    FRAMES = 1000  # 500

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

    a = torch.linspace(0, ALPHA_RANGE, FRAMES)
    for i, a in tqdm(enumerate(a), total=len(a), desc="frames"):
        params = params1 + a * (params2 - params1)
        set_params(
            model,
            unflatten_params(params, params_shape),
        )

        # prediction with combined normalization
        z_pred, _ = predict_rgb(mlp_input, norm)
        z_pred = z_pred.clamp(0, 1)  # will sometimes exceed

        folder = f"expressivity_{activation_id}_{HIDDEN_LAYERS}_{NEURONS}"
        (ANIMATION_DIR / folder).mkdir(parents=True, exist_ok=True)
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
