import argparse
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from analog_rnn_fixture import (
    DATASET,
    MATERIAL,
    RESOLUTION,
    N,
    cross_entropy,
    load_source,
    probabilities,
    region,
    sensors,
    sim,
    to_node,
)
from cuwave.sensitivity import reconstruction_sensitivity
from cuwave.utils import threshold
from cuwave.wave import simulate

from postprocessing import save_csv, save_temp_fig
from solvers.optimization import DensityFilter, dprojection, projection

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# geometry: a trainable material square centered in x spanning the full height
DESIGN_X = (50.0, 150.0)  # wider (20, 180) plateaued at 0.93 accuracy for 5/class

# optimization
SAMPLES_PER_CLASS = 4  # -1 for every clip the dataset holds
BATCH_SIZE = 1
EPOCHS = 200
LR = 2e-2
RMIN = 2.0
AMPLITUDE_PENALTY = 0.1  # 0.0 -> pure cross-entropy -log p[label]

# projection: beta grows slowly (per epoch) so the medium binarizes late in training
ETA = 0.5
BETA0 = 1.0
BETA_GROWTH = 2.0
BETA_STEP = 16
BETA_MAX = 64.0

# evaluation
THRESHOLD = 0.5  # the projection maps onto [0, 1], so its midpoint is the cut


# -------------------------------------- helper ---------------------------------------
def field_fig():
    # borderless axes at one pixel per node, the shape every frame and figure shares
    fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def physical(xval, beta):
    full = cp.zeros(sim.Nx_padded, dtype=sim.dtype).ravel()
    full[active] = cp.asarray(xval.ravel(), dtype=sim.dtype)
    x_tilde = density_filter(full.reshape(sim.Nx_padded))
    return projection(x_tilde, beta, ETA) * design, x_tilde


# --------------------------------------- setup ---------------------------------------
# design region: the trainable material square
design = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
x_lo, x_hi = to_node((DESIGN_X[0], 0.0))[0], to_node((DESIGN_X[1], 0.0))[0]
design[x_lo:x_hi, region[1]] = 1.0
active = cp.where(design.ravel() > 0)[0]

density_filter = DensityFilter(RMIN, sim.Nx_padded, xp=cp, dtype=sim.dtype)
beta_of = lambda epoch: min(BETA0 * BETA_GROWTH ** (epoch // BETA_STEP), BETA_MAX)

# ------------------------------------- load data -------------------------------------
data = np.load(DATASET)
labels = data["y"]
classes = data["classes"]
clips = data["X"]

if SAMPLES_PER_CLASS != -1:
    keep = np.concatenate(
        [np.where(labels == c)[0][:SAMPLES_PER_CLASS] for c in range(len(classes))]
    )
    labels = labels[keep]
    clips = clips[keep]

sources = [load_source(clip) for clip in clips]
samples = len(sources)

# inverse-frequency class weights (mean weight 1) counter the imbalanced clip counts
counts = np.maximum(np.bincount(labels, minlength=len(classes)), 1)
class_weights = samples / (len(classes) * counts)
print(f"{samples} clips over {len(classes)} classes, {int(design.sum())} design nodes")

# ------------------------------------ optimization -----------------------------------
n = int(active.size)
x = (0.5 + 0.1 * (torch.rand(n, device=device) - 0.5)).requires_grad_(True)
optimizer = torch.optim.Adam([x], lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

loss_history = []
acc_history = []
grad_scale = None

ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/analog_rnn"
if args.animate:
    (ANIMATION_DIR / "optimization").mkdir(parents=True, exist_ok=True)
tic = time.time()
for epoch in range(EPOCHS):
    beta = beta_of(epoch)
    perm = torch.randperm(samples).tolist()

    loss_sum = 0.0
    correct = 0
    for start in range(0, samples, BATCH_SIZE):
        batch = perm[start : start + BATCH_SIZE]
        gamma, x_tilde = physical(x.detach().cpu().numpy(), beta)
        d_mass, d_stiff = sim.parametrization_jacobian(gamma)

        grad_gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
        for i in batch:
            label = int(labels[i])
            objective = cross_entropy(label, AMPLITUDE_PENALTY)
            loss, grads, um, _ = reconstruction_sensitivity(
                sim, sources[i], gamma, sensors.nodes, sensors.objective(objective)
            )
            w = float(class_weights[label])
            grad_gamma += w * (d_mass * grads["mass"] + d_stiff * grads["stiff"])
            loss_sum += w * loss
            correct += int(probabilities(sensors.traces(um)).argmax() == label)
        grad_gamma /= len(batch)

        dpx = dprojection(x_tilde, beta, ETA) * design
        grad_obj = density_filter.adjoint(grad_gamma * dpx).ravel()[active].get()

        # frozen after the first batch, so the step never tracks the loss scale
        grad_scale = np.abs(grad_obj).max() if grad_scale is None else grad_scale
        x.grad = torch.as_tensor(grad_obj / grad_scale, dtype=x.dtype, device=device)
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
    scheduler.step()

    loss_history.append(loss_sum / samples)
    acc_history.append(correct / samples)
    print(
        f"epoch {epoch}: loss {loss_history[-1]:.4f}  "
        f"accuracy {acc_history[-1]:.2f}  beta {beta:.0f}"
    )

    if args.animate:
        fig, ax = field_fig()
        ax.imshow(gamma.get()[region].T, origin="lower", cmap="binary", vmin=0, vmax=1)
        fig.savefig(ANIMATION_DIR / "optimization" / f"frame_{epoch:04d}.jpg")
        plt.close()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / EPOCHS:.2f} s/epoch)")

# ----------------------------------- postprocessing ----------------------------------
gamma_final, _ = physical(x.detach().cpu().numpy(), beta_of(EPOCHS))
final = threshold(gamma_final, THRESHOLD) * design

# the thresholded design is the one that can be built, so it is the one that is reported
confusion = np.zeros((len(classes), len(classes)), dtype=int)
for source, label in zip(sources, labels):
    um = simulate(sim, source, final, sensors=sensors.nodes)[1]
    probs = probabilities(sensors.traces(um)).get()
    confusion[int(label), int(probs.argmax())] += 1
    print(f"true {classes[int(label)]:8s} probs {np.round(probs, 3)}")
accuracy = np.trace(confusion) / confusion.sum()
print(f"thresholded accuracy {accuracy:.3f}")
print(confusion)

design_view = final.get()[region]

# --------------------------------------- export --------------------------------------
# binarized medium, reusable as a material mask by the evaluation driver
material = design_view > THRESHOLD
np.save(MATERIAL, material)
print(f"saved binarized material to {MATERIAL}")

if args.animate:
    wave_dir = ANIMATION_DIR / "wavefield"
    wave_dir.mkdir(parents=True, exist_ok=True)
    _, snaps = simulate(sim, sources[0], final, record_every=max(1, N // 200))
    scale = float(np.max(np.abs(snaps)))
    overlay = np.ma.masked_where(design_view < THRESHOLD, design_view)
    for f, snap in enumerate(snaps):
        fig, ax = field_fig()
        ax.imshow(snap[region].T, origin="lower", cmap="seismic", vmin=-scale, vmax=scale)
        ax.imshow(overlay.T, origin="lower", cmap="binary", vmin=0, vmax=1, alpha=0.85)
        fig.savefig(wave_dir / f"frame_{f:04d}.jpg")
        plt.close()
# -------------------------------- book postprocessing --------------------------------
elif args.book:
    save_csv(
        CSV_DIR / "analog_rnn_history.csv",
        epoch=np.arange(EPOCHS),
        loss=np.array(loss_history),
        accuracy=np.array(acc_history),
    )
    save_csv(CSV_DIR / "analog_rnn_confusion.csv", confusion=confusion.ravel())

    fig, ax = field_fig()
    ax.imshow(design_view.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    fig.savefig(RGB_PDF_DIR / "analog_rnn_design.pdf", transparent=True)
    plt.close()
else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].plot(loss_history, "k")
    axes[0, 1].plot(acc_history, "k")
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].imshow(design_view.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(confusion, origin="upper", cmap="cividis")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
    save_temp_fig(RESULTS_DIR / "analog_rnn")
    plt.show()
