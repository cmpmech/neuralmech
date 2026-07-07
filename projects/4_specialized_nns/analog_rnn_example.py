import argparse
import math
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, resample, sosfiltfilt

from postprocessing import save_csv
from solvers.wave import (
    acoustic_simulation,
    build_sponge,
    compile_kernels,
    define_excitation,
    define_homogeneous_Neumann_BC,
    define_step_method,
    setup_source,
)
from solvers.wave_sensitivity import compute_sensitivity_classification

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# geometry: source on the left wall, three probes on the right wall (one per class),
# a trainable material square centered in x spanning the full height
LENGTHS = (200.0, 100.0)
SOURCE_CENTER = (10.0, 50.0)
DESIGN_X = (50.0, 150.0)
PROBE_X = 190.0
PROBE_Y = (25.0, 50.0, 75.0)

# discretization
# RESOLUTION = (96, 48)
RESOLUTION = (200, 100)
CFL = 0.9  # 0.5
T = 1.5

# physics: air (material 1) and a moderate-contrast dense scatterer (material 2)
RHO1, RHO2 = 1.204, 12.04
KAPPA1, KAPPA2 = 1.419e5, 1.419e5
AMPLITUDE = 1e3
POINTS_PER_WAVELENGTH = 10

# absorbing sponge on every edge [x-, x+, y-, y+] so probe energies are not degenerate
BOUNDARIES = ["pml", "pml", "pml", "pml"]
SPONGE_WIDTH = 8  # 50  # 8
SPONGE_BETA = 1.5  # 0.1  # 1.5

# optimization
SAMPLES_PER_CLASS = 2  # first clips kept per class for overfitting (-1 uses all)
EPOCHS = 300
LR = 5e-2
RMIN = 2.0

# projection: beta grows slowly (per epoch) so the medium binarizes late in training
ETA = 0.5
BETA0 = 1.0
BETA_GROWTH = 2.0
BETA_STEP = 16  # epochs between beta updates
BETA_MAX = 1.0


# -------------------------------------- helper ---------------------------------------
def density_filter(x):
    return ndi.convolve(x, KERNEL, mode="constant", cval=0.0) / HS


def filter_adjoint(g):
    return ndi.convolve(g / HS, KERNEL, mode="constant", cval=0.0)


def projection(x, beta, eta):
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return (a + cp.tanh(beta * (x - eta))) / (a + b)


def dprojection(x, beta, eta):
    a, b = math.tanh(beta * eta), math.tanh(beta * (1.0 - eta))
    return beta * (1.0 - cp.tanh(beta * (x - eta)) ** 2) / (a + b)


def physical(xval, beta):
    full = cp.zeros(sim.Nx_padded, dtype=sim.dtype).ravel()
    full[active] = cp.asarray(xval.ravel(), dtype=sim.dtype)
    x_tilde = density_filter(full.reshape(sim.Nx_padded))
    return projection(x_tilde, beta, ETA) * design, x_tilde


def load_source(clip):
    # resample the clip onto the simulation time base, low-pass to the grid's max
    # resolvable frequency (zero-phase), normalize, and scale to the source amplitude
    sos = butter(8, f_max, btype="low", fs=1.0 / dt, output="sos")
    wave = sosfiltfilt(sos, resample(clip, N))
    wave = wave / np.max(np.abs(wave))
    return cp.asarray(AMPLITUDE * wave[:, None] / np.prod(dx), dtype=sim.dtype)


def simulate_frames(gamma, signal, record_every):
    mat = sim.build_materials(gamma, sponge)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source_position, kernels, mat)
    U = cp.zeros((2, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1 = U[0], U[1]
    frames = []
    for t in range(sim.N):
        u0 = fd_step(u0, u1, u0)
        u0 = excitation_step(u0, signal, t)
        u0 = bc_step(u0)
        u1, u0 = u0, u1
        if t % record_every == 0:
            frames.append(u1[crop].get())
    return frames


# --------------------------------------- setup ---------------------------------------
# increase grid by sponge layer on each absorbing edge
pad_lo = tuple(SPONGE_WIDTH if BOUNDARIES[2 * d] == "pml" else 0 for d in range(2))
pad_hi = tuple(SPONGE_WIDTH if BOUNDARIES[2 * d + 1] == "pml" else 0 for d in range(2))

# spatial grid
Nx = tuple(RESOLUTION[d] + pad_lo[d] + pad_hi[d] for d in range(2))
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(2))
# helpers
to_index = lambda coord: tuple(
    int(round(coord[d] / dx[d])) + pad_lo[d] for d in range(2)
)
crop = tuple(slice(1 + pad_lo[d], Nx[d] - 1 - pad_hi[d]) for d in range(2))

# temporal grid
wavespeeds = np.sqrt(np.array([KAPPA2 / RHO2, KAPPA1 / RHO1]))
dt = CFL * min(dx) / np.max(wavespeeds) / math.sqrt(2)
N = math.ceil(T / dt)

f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))
print(f"steps {N}, max resolvable frequency {f_max:.1f} Hz")

sim = acoustic_simulation(
    Nx,
    dx,
    N,
    dt,
    (4, 64),
    precision="float32",
    rho1=RHO1,
    rho2=RHO2,
    kappa1=KAPPA1,
    kappa2=KAPPA2,
)

# boundary conditions
EDGES = [(0, "lo"), (0, "hi"), (1, "lo"), (1, "hi")]
sides = [e for e, b in zip(EDGES, BOUNDARIES) if b == "pml"]
sponge = build_sponge(
    sim, SPONGE_WIDTH, 2.0 * SPONGE_BETA / (KAPPA1 * dt), power=3, sides=sides
)

# source and probes: probe m is the class-m readout (hostile 0, neutral 1, passive 2)
src_i, src_j = to_index(SOURCE_CENTER)
source_position = cp.array([[src_i], [src_j]], dtype=cp.int32)
probes = [to_index((PROBE_X, y)) for y in PROBE_Y]
sensors = cp.array([[p[0] for p in probes], [p[1] for p in probes]], dtype=cp.int32)

# design region: the trainable material square
design = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
x_lo, x_hi = to_index((DESIGN_X[0], 0))[0], to_index((DESIGN_X[1], 0))[0]
y_lo, y_hi = 1 + pad_lo[1], Nx[1] - 1 - pad_hi[1]
design[x_lo:x_hi, y_lo:y_hi] = 1.0
active = cp.where(design.ravel() > 0)[0]

# conic density filter
ceil_r = int(math.ceil(RMIN))
ki, kj = cp.meshgrid(
    cp.arange(-ceil_r, ceil_r + 1), cp.arange(-ceil_r, ceil_r + 1), indexing="ij"
)
KERNEL = cp.maximum(0.0, RMIN - cp.sqrt(ki**2 + kj**2)).astype(sim.dtype)
HS = ndi.convolve(
    cp.ones(sim.Nx_padded, dtype=sim.dtype), KERNEL, mode="constant", cval=0.0
)

beta_of = lambda epoch: min(BETA0 * BETA_GROWTH ** (epoch // BETA_STEP), BETA_MAX)

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "minecraft_mobs.npz")
if data["X"].shape[0] == 0:
    raise SystemExit("empty minecraft_mobs.npz; run minecraft_mobs_download.py first")
labels = data["y"]
classes = data["classes"]
clips = data["X"]

if SAMPLES_PER_CLASS != -1:
    keep = np.concatenate([np.where(labels == c)[0][:SAMPLES_PER_CLASS]
                           for c in range(len(classes))])
    labels = labels[keep]
    clips = clips[keep]

signals = [load_source(clip) for clip in clips]
samples = len(signals)

# inverse-frequency class weights (mean weight 1) counter the imbalanced clip counts
counts = np.bincount(labels, minlength=len(classes))
class_weights = samples / (len(classes) * counts)

# ------------------------------------ optimization -----------------------------------
n = int(active.size)
x = (0.5 + 0.1 * (torch.rand(n, device=device) - 0.5)).requires_grad_(True)
optimizer = torch.optim.Adam([x], lr=LR)

loss_history = []
acc_history = []
grad_scale = None

ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/analog_rnn"
if args.animate:
    (ANIMATION_DIR / "optimization").mkdir(parents=True, exist_ok=True)
tic = time.time()
for epoch in range(EPOCHS):
    beta = beta_of(epoch)
    gamma, x_tilde = physical(x.detach().cpu().numpy(), beta)

    grad_gamma = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    loss_sum = 0.0
    correct = 0
    for signal, label in zip(signals, labels):
        w = float(class_weights[label])
        source = setup_source(source_position, signal)
        loss, probs, grad = compute_sensitivity_classification(
            sim, gamma, source, sensors, sponge, int(label)
        )
        grad_gamma += w * grad
        loss_sum += w * loss
        correct += int(probs.argmax() == label)
    grad_gamma /= samples

    dpx = dprojection(x_tilde, beta, ETA) * design
    grad_obj = filter_adjoint(grad_gamma * dpx).ravel()[active].get()

    grad_scale = np.abs(grad_obj).max() if grad_scale is None else grad_scale
    x.grad = torch.as_tensor(grad_obj / grad_scale, dtype=x.dtype, device=device)
    optimizer.step()
    with torch.no_grad():
        x.clamp_(0.0, 1.0)

    loss_history.append(loss_sum / samples)
    acc_history.append(correct / samples)
    print(
        f"epoch {epoch}: loss {loss_history[-1]:.4f}  "
        f"accuracy {acc_history[-1]:.2f}  beta {beta:.0f}"
    )

    if args.animate:
        fig, ax = plt.subplots(
            figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150
        )
        ax.imshow(gamma.get()[crop].T, origin="lower", cmap="binary", vmin=0, vmax=1)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(ANIMATION_DIR / "optimization" / f"frame_{epoch:04d}.jpg")
        plt.close()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / EPOCHS:.2f} s/epoch)")

# ----------------------------------- postprocessing ----------------------------------
gamma_final, _ = physical(x.detach().cpu().numpy(), beta_of(EPOCHS))

# final predictions and confusion matrix (rows true class, columns predicted class)
confusion = np.zeros((len(classes), len(classes)), dtype=int)
for signal, label in zip(signals, labels):
    source = setup_source(source_position, signal)
    _, probs, _ = compute_sensitivity_classification(
        sim, gamma_final, source, sensors, sponge, int(label)
    )
    confusion[int(label), int(probs.argmax())] += 1
    print(f"true {classes[int(label)]:8s} probs {np.round(probs, 3)}")
accuracy = np.trace(confusion) / confusion.sum()
print(f"training accuracy {accuracy:.3f}")
print(confusion)

design_view = gamma_final.get()[crop]

if args.animate:
    wave_dir = ANIMATION_DIR / "wavefield"
    wave_dir.mkdir(parents=True, exist_ok=True)
    frames = simulate_frames(gamma_final, signals[0], record_every=max(1, N // 200))
    scale = max(float(np.max(np.abs(f))) for f in frames)
    overlay = np.ma.masked_where(design_view < 0.5, design_view)
    for f, frame in enumerate(frames):
        fig, ax = plt.subplots(
            figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150
        )
        ax.imshow(frame.T, origin="lower", cmap="seismic", vmin=-scale, vmax=scale)
        ax.imshow(overlay.T, origin="lower", cmap="binary", vmin=0, vmax=1, alpha=0.85)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
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

    fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150)
    ax.imshow(design_view.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RGB_PDF_DIR / "analog_rnn_design.pdf", transparent=True)
    plt.close()
else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].semilogy(loss_history, "k")
    axes[0, 1].plot(acc_history, "k")
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].imshow(design_view.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(confusion, origin="upper", cmap="cividis")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
    plt.show()
