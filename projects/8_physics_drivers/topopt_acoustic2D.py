import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import mmapy
import numpy as np
import torch
from cuwave.boundary import pad_for_sponge, sponge
from cuwave.geometry import box, nodes
from cuwave.sensitivity import reconstruction_sensitivity
from cuwave.signals import sineburst
from cuwave.utils import energy, point_source, response, response_gradient
from cuwave.wave import AcousticWave, grid_coords, simulate, stable_dt

from solvers.optimization import DensityFilter, dprojection, projection

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# geometry (metres)
LENGTHS = (18.0, 9.0)
SOURCE_CENTER = (0.3, 4.5)
CHANNEL_HEIGHT = 2.0
DESIGN_X0 = 6.0
TARGET_CENTER = (17.0, 4.5)
TARGET_SIZE = (2.0, 2.0)

# discretization
RESOLUTION = (128, 64)
SPACE_ORDER = 2  # finite difference order, any even number
SAFETY = 0.5  # fraction of the stable time step
THREADS = (4, 64)

# physics
RHO1, RHO2 = 1.204, 2643.0
KAPPA1, KAPPA2 = 1.419e5, 6.87e8
FREQUENCY = 200.0
POINTS_PER_WAVELENGTH = 10
CYCLES = 3
AMPLITUDE = 1e3
T = 0.18

# objective
AMPLIFY = False

# boundary conditions per edge [x-, x+, y-, y+]: "reflective" or "sponge"
# BOUNDARIES = ["reflective", "reflective", "reflective", "reflective"]
# BOUNDARIES = ["reflective", "reflective", "sponge", "sponge"]
BOUNDARIES = ["sponge", "sponge", "sponge", "sponge"]

# absorbing sponge on the "sponge" edges
SPONGE_THICKNESS = 1.75  # metres, so it does not need rescaling with RESOLUTION
SPONGE_BETA = 1.5  # peak damping d * dt / 2m at the wall

# optimization
USE_ADAM = False  # Adam instead of MMA
EPOCHS = 160
LR = 1e-2  # Adam step size
MMA_MOVE = 0.2  # MMA step move limit
RMIN = 2.0
ETA = 0.5
BETA0 = 1.0
BETA_GROWTH = 2.0
BETA_STEP = 18  # epochs between beta updates
BETA_MAX = 64.0


# -------------------------------------- helper ---------------------------------------
def physical(xval, beta):
    full = cp.zeros(sim.Nx_padded, dtype=sim.dtype).ravel()
    full[active] = cp.asarray(xval.ravel(), dtype=sim.dtype)
    x_tilde = density_filter(full.reshape(sim.Nx_padded))
    return projection(x_tilde, beta, ETA) * design, x_tilde


# --------------------------------------- setup ---------------------------------------
# spatial grid, grown by a sponge layer behind each absorbing edge
dx = tuple(LENGTHS[d] / (RESOLUTION[d] - 3) for d in range(2))
faces = [f for f, b in enumerate(BOUNDARIES) if b == "sponge"]
Nx, width, origin, region = pad_for_sponge(RESOLUTION, dx, SPONGE_THICKNESS, faces)
x0, y0 = origin  # where the region of interest starts, the sponge sitting before it

# temporal grid
wavespeeds = np.sqrt(np.array([KAPPA2 / RHO2, KAPPA1 / RHO1]))
dt = SAFETY * stable_dt(dx, np.max(wavespeeds), SPACE_ORDER)
N = math.ceil(T / dt)

f_max = np.min(wavespeeds) / (POINTS_PER_WAVELENGTH * max(dx))
print(f"max resolvable frequency {f_max:.1f} Hz")
assert FREQUENCY <= f_max, (
    f"FREQUENCY {FREQUENCY:.0f} Hz exceeds the max resolvable {f_max:.1f} Hz; "
    f"lower FREQUENCY or raise RESOLUTION"
)

sim = AcousticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision="float32",
    space_order=SPACE_ORDER,
    rho1=RHO1,
    rho2=RHO2,
    kappa1=KAPPA1,
    kappa2=KAPPA2,
)
# damping sets a compile flag, so it has to be in place before any kernel compiles
if faces:
    air = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    sim = replace(sim, damping=sponge(sim, air, width, SPONGE_BETA, faces))

# source (sine burst)
t = np.linspace(0, (N - 1) * dt, N)
source_coords = (x0 + SOURCE_CENTER[0], y0 + SOURCE_CENTER[1])
source = point_source(sim, source_coords, sineburst(t, AMPLITUDE, FREQUENCY, CYCLES))

# target box: objective
coords = grid_coords(Nx, dx, dtype=sim.dtype)
target_center = (x0 + TARGET_CENTER[0], y0 + TARGET_CENTER[1])
sensors = nodes(box(coords, target_center, TARGET_SIZE))

# design region: two blocks filling the channel walls right of DESIGN_X0
block_x = (DESIGN_X0 + LENGTHS[0]) / 2, LENGTHS[0] - DESIGN_X0
block_center = (x0 + block_x[0], y0 + LENGTHS[1] / 2)
blocks = box(coords, block_center, (block_x[1], LENGTHS[1]))
channel = box(coords, block_center, (block_x[1], CHANNEL_HEIGHT))
design = (blocks & ~channel).astype(sim.dtype)
design_area = float(design.sum())
active = cp.where(design.ravel() > 0)[0]  # flat indices of the design variables
print(f"{sensors.shape[1]} target nodes over {int(design_area)} design nodes")

# conic density filter
density_filter = DensityFilter(RMIN, sim.Nx_padded, xp=cp, dtype=sim.dtype)

# beta continuation
beta_of = lambda epoch: min(BETA0 * BETA_GROWTH ** (epoch // BETA_STEP), BETA_MAX)

name = "topopt_acoustic2D" + ("_adam" if USE_ADAM else "")
name += "_amplify" if AMPLIFY else ""
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames" / name

# ------------------------------------ optimization -----------------------------------
objective = energy(sim)
n = int(active.size)
xval = np.full((n, 1), 0.5)
if USE_ADAM:
    x = torch.full((n,), 0.5, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=LR)
else:
    xold1, xold2 = xval.copy(), xval.copy()
    low, upp = np.zeros((n, 1)), np.ones((n, 1))
    xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
    a0, a_mma, c_mma, d_mma = 1.0, np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

energy_air = response(sim, source, cp.zeros(sim.Nx_padded, dtype=sim.dtype),
                      sensors, objective)[0]
cost_ref = None
cost_history = []

if args.animate:
    (ANIMATION_DIR / "optimization").mkdir(parents=True, exist_ok=True)
tic = time.time()
for epoch in range(EPOCHS):
    beta = beta_of(epoch)
    if USE_ADAM:
        xval = x.detach().cpu().numpy()
    gamma, x_tilde = physical(xval, beta)

    cost, grad_gamma = response_gradient(
        sim, source, gamma, sensors, objective, adjoint=reconstruction_sensitivity
    )

    dpx = dprojection(x_tilde, beta, ETA) * design
    grad_obj = density_filter.adjoint(grad_gamma * dpx).ravel()[active].get()

    cost_ref = cost if cost_ref is None else cost_ref
    sign = -1.0 if AMPLIFY else 1.0
    if USE_ADAM:
        x.grad = torch.as_tensor(sign * grad_obj / cost_ref, dtype=x.dtype, device=device)
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
    else:
        f0val = sign * cost / cost_ref
        df0dx = (sign * grad_obj / cost_ref).reshape(n, 1)
        fval = np.zeros((0, 1))
        dfdx = np.zeros((0, n))

        xmma, _, _, _, _, _, _, _, _, low, upp = mmapy.mmasub(
            0, n, epoch + 1, xval, xmin, xmax, xold1, xold2,
            f0val, df0dx, fval, dfdx, low, upp,
            a0, a_mma, c_mma, d_mma, move=MMA_MOVE,
        )
        xold2, xold1 = xold1, xval
        xval = xmma

    cost_history.append(cost / energy_air)
    print(
        f"epoch {epoch}: reduction {10 * math.log10(cost / energy_air):.2f} dB  "
        f"beta {beta:.0f}"
    )

    if args.animate:
        fig, ax = plt.subplots(
            figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150
        )
        ax.imshow(
            gamma.get()[region].T,
            origin="lower",
            cmap="binary",
            vmin=0,
            vmax=1,
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(ANIMATION_DIR / "optimization" / f"frame_{epoch:04d}.jpg")
        plt.close()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s  ({(toc - tic) / EPOCHS:.2f} s/epoch)")

# ----------------------------------- postprocessing ----------------------------------
if USE_ADAM:
    xval = x.detach().cpu().numpy()
gamma_final, _ = physical(xval, beta_of(EPOCHS))
gamma_binary = (gamma_final > 0.5).astype(sim.dtype) * design

energy_design = response(sim, source, gamma_final, sensors, objective)[0]
energy_binary = response(sim, source, gamma_binary, sensors, objective)[0]
db = lambda e: 10 * math.log10(e / energy_air)
print(
    f"air baseline   {db(energy_air):.2f} dB\n"  # energy_air is reference -> 0 dB
    f"design         {db(energy_design):.2f} dB\n"
    f"binarized      {db(energy_binary):.2f} dB  "
    f"(volume fraction {float(gamma_binary.sum() / design_area):.3f})"
)

design = gamma_binary.get()[region]

if args.animate:
    wave_dir = ANIMATION_DIR / "wavefield"
    wave_dir.mkdir(parents=True, exist_ok=True)
    _, snaps = simulate(sim, source, gamma_binary, record_every=max(1, N // 200))
    scale = float(np.max(np.abs(snaps)))
    overlay = np.ma.masked_where(design < 0.5, design)
    for f, snap in enumerate(snaps):
        fig, ax = plt.subplots(
            figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150
        )
        ax.imshow(snap[region].T, origin="lower", cmap="seismic", vmin=-scale, vmax=scale)
        ax.imshow(overlay.T, origin="lower", cmap="binary", vmin=0, vmax=1, alpha=0.85)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(wave_dir / f"frame_{f:04d}.jpg")
        plt.close()
elif args.book:
    fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150)
    ax.imshow(design.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RGB_PDF_DIR / f"{name}.pdf", transparent=True)
    plt.close()
else:
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    axes[0].semilogy(cost_history, "k")
    axes[1].imshow(design.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
