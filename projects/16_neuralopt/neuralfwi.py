import argparse
import json
import subprocess
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import torch
from cuwave.evals import f1_score, l2_error
from cuwave.nn import Generator
from cuwave.optimization import Adam
from cuwave.regularization import Projection
from cuwave.utils import interior_slice, misfit, misfit_gradient

from helper import GAMMA_MIN, FWI, UNet, normalize

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ansatz", choices=["voxel", "projection", "neural", "pretrained"], default="voxel"
)
parser.add_argument("--theme", choices=["light", "dark"], default="light")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
ITERS = 60
LR = {"voxel": 1e-1, "projection": 1e-1, "neural": 3e-3, "pretrained": 1e-4}
LR = LR[args.ansatz]

# model settings
CHANNELS = [16, 16, 16, 16, 8, 1]  # generator of the neural ansatz
KERNEL = 5
OUTPUT_BIAS = 3.0
BETA, ETA = 3.0, 0.5  # sharpness and threshold of the projection ansatz
UNET_CHANNELS = [16, 32, 64, 128]  # pretrained ansatz, see neuralfwi_pretraining.py

# postprocessing
FPS = 10
cmap = "hot" if args.theme == "light" else "hot_r"  # dark: white holes on black
name = f"neuralfwi_{args.ansatz}_{args.theme}"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames" / name

# ---------------------------------------- setup --------------------------------------
fwi = FWI()
sim = fwi.sim
truth = fwi.circles()
observed = fwi.observe(truth)
interior = interior_slice(sim)
homogeneous = cp.ones(sim.Nx_padded, dtype=sim.dtype)
initial_misfit = misfit(sim, fwi.sources, homogeneous, fwi.sensors, observed)
gradient_of = lambda gamma: misfit_gradient(
    sim, fwi.sources, gamma, fwi.sensors, observed
)

to_cupy = lambda field: cp.asarray(field.detach())[0, 0]
projection = Projection(BETA, ETA)
chain = lambda gradient: torch.as_tensor((1.0 - GAMMA_MIN) * gradient)[None, None]
if args.ansatz == "neural":
    model = Generator(CHANNELS, sim.Nx_padded, kernel=KERNEL, output_bias=OUTPUT_BIAS)
    model = model.to(device)
    forward = lambda: model()
elif args.ansatz == "pretrained":
    model = UNet(UNET_CHANNELS).to(device)
    state = torch.load(MODEL_DIR / "neuralfwi_unet.pt", map_location=device)
    model.load_state_dict(state)
    # the network sees the first gradient, the one every ansatz starts from
    _, gradient = gradient_of(homogeneous)
    first_gradient = normalize(torch.as_tensor(gradient)[None, None])
    forward = lambda: torch.sigmoid(model(first_gradient))
if args.ansatz in ("voxel", "projection"):
    x = homogeneous.copy()
    optimizer = Adam(lr=LR)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def render(gamma, path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
    ax.imshow(gamma[interior].get().T, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path)
    plt.close()


# ------------------------------------ optimization -----------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
history = []
tic = time.time()
for iteration in range(ITERS + 1):
    if args.ansatz == "voxel":
        gamma = x
    elif args.ansatz == "projection":
        gamma = GAMMA_MIN + (1.0 - GAMMA_MIN) * projection(x)
    else:
        field = forward()
        gamma = GAMMA_MIN + (1.0 - GAMMA_MIN) * to_cupy(field)
    if args.animate:
        render(gamma, ANIMATION_DIR / f"frame_{iteration}.jpg")
    cost, gradient = gradient_of(gamma)
    history.append(cost)
    if iteration == ITERS:
        break
    if args.ansatz == "voxel":
        x = cp.clip(optimizer.step(x, gradient), GAMMA_MIN, 1.0)
    elif args.ansatz == "projection":
        gradient = projection.grad(x, (1.0 - GAMMA_MIN) * gradient)
        x = cp.clip(optimizer.step(x, gradient), 0.0, 1.0)
    else:
        optimizer.zero_grad()
        field.backward(chain(gradient))
        optimizer.step()
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
reference, recovered = truth[interior], gamma[interior]
metrics = dict(
    misfit=history[-1] / initial_misfit,
    l2=float(l2_error(recovered, reference)),
    f1=float(f1_score(recovered, reference, threshold=0.5)),
)
print(", ".join(f"{k} {v:.2e}" for k, v in metrics.items()))

render(truth, RESULTS_DIR / f"neuralfwi_truth_{args.theme}.png")
render(gamma, RESULTS_DIR / f"{name}.png")
(RESULTS_DIR / f"{name}.json").write_text(json.dumps(dict(history=history, **metrics)))
if args.animate:
    frames = str(ANIMATION_DIR / "frame_%d.jpg")
    video = str(RESULTS_DIR / f"{name}.mp4")
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS)]
    subprocess.run(ffmpeg + ["-i", frames, "-pix_fmt", "yuv420p", video], check=True)
