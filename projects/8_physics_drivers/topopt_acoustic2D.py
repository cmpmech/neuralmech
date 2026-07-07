import argparse
import math
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.ndimage as ndi
import matplotlib.pyplot as plt
import mmapy
import numpy as np

from solvers.wave import (
    acoustic_simulation,
    build_sponge,
    compile_kernels,
    define_excitation,
    define_homogeneous_Neumann_BC,
    define_step_method,
    setup_source,
    simulate,
)
from solvers.wave_sensitivity import compute_sensitivity_pml

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

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
CFL = 0.5

# physics:
RHO1, RHO2 = 1.204, 2643.0
KAPPA1, KAPPA2 = 1.419e5, 6.87e8
FREQUENCY = 200.0
POINTS_PER_WAVELENGTH = 10
CYCLES = 3
AMPLITUDE = 1e3
T = 0.18

# objective
AMPLIFY = False

# boundary conditions per edge [x-, x+, y-, y+]: "reflective" or "pml"
# BOUNDARIES = ["reflective", "reflective", "reflective", "reflective"]
# BOUNDARIES = ["reflective", "reflective", "pml", "pml"]
BOUNDARIES = ["pml", "pml", "pml", "pml"]

# absorbing sponge on the "pml" edges
SPONGE_WIDTH = 12
SPONGE_BETA = 1.5

# optimization
EPOCHS = 160
RMIN = 2.0
ETA = 0.5
MMA_MOVE = 0.2
BETA0 = 1.0
BETA_GROWTH = 2.0
BETA_STEP = 18  # epochs between beta updates
BETA_MAX = 64.0


# -------------------------------------- helper ---------------------------------------
def sineburst(t, amplitude, frequency, cycles):
    mask = (t > 0) & (t <= cycles / frequency)
    return (
        amplitude
        * mask
        * np.sin(2 * np.pi * frequency * t)
        * np.sin(np.pi * frequency * t / cycles) ** 2
    )


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


def target_energy(gamma):
    _, um = simulate(sim, source, gamma, damping=sponge, sensors=sensors)
    return 0.5 * float(np.prod(sim.dx)) * sim.dt * float(cp.sum(um**2))


def simulate_frames(gamma, record_every):
    mat = sim.build_materials(gamma, sponge)
    kernels = compile_kernels(sim)
    fd_step = define_step_method(sim, kernels, mat)
    bc_step = define_homogeneous_Neumann_BC(sim, kernels)
    excitation_step = define_excitation(sim, source.position, kernels, mat)
    U = cp.zeros((2, *sim.Nx_padded), dtype=sim.dtype)
    u0, u1 = U[0], U[1]
    frames = []
    for t in range(sim.N):
        u0 = fd_step(u0, u1, u0)
        u0 = excitation_step(u0, source.signal, t)
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
print(f"max resolvable frequency {f_max:.1f} Hz")
assert FREQUENCY <= f_max, (
    f"FREQUENCY {FREQUENCY:.0f} Hz exceeds the max resolvable {f_max:.1f} Hz; "
    f"lower FREQUENCY or raise RESOLUTION"
)

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

# source (sine burst)
t = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)
src_i, src_j = to_index(SOURCE_CENTER)
source = setup_source(cp.array([[src_i], [src_j]], dtype=cp.int32), signal)

# target box: objective
lo_i, lo_j = to_index(
    (TARGET_CENTER[0] - TARGET_SIZE[0] / 2, TARGET_CENTER[1] - TARGET_SIZE[1] / 2)
)
hi_i, hi_j = to_index(
    (TARGET_CENTER[0] + TARGET_SIZE[0] / 2, TARGET_CENTER[1] + TARGET_SIZE[1] / 2)
)
box_i, box_j = cp.meshgrid(
    cp.arange(lo_i, hi_i + 1), cp.arange(lo_j, hi_j + 1), indexing="ij"
)
sensors = cp.stack([box_i.ravel(), box_j.ravel()]).astype(cp.int32)
um = cp.zeros((N, sensors.shape[1]), dtype=sim.dtype)

# design region
design = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
x0 = to_index((DESIGN_X0, 0))[0]
ch_lo = to_index((0, LENGTHS[1] / 2 - CHANNEL_HEIGHT / 2))[1]
ch_hi = to_index((0, LENGTHS[1] / 2 + CHANNEL_HEIGHT / 2))[1]
x_hi = Nx[0] - 1 - pad_hi[0]
y_lo = 1 + pad_lo[1]
y_hi = Nx[1] - 1 - pad_hi[1]
design[x0:x_hi, y_lo:ch_lo] = 1.0  # lower block
design[x0:x_hi, ch_hi:y_hi] = 1.0  # upper block
design_area = float(design.sum())
active = cp.where(design.ravel() > 0)[0]  # flat indices of the design variables

# optimization helpers
# conic density filter
ceil_r = int(math.ceil(RMIN))
ki, kj = cp.meshgrid(
    cp.arange(-ceil_r, ceil_r + 1), cp.arange(-ceil_r, ceil_r + 1), indexing="ij"
)
KERNEL = cp.maximum(0.0, RMIN - cp.sqrt(ki**2 + kj**2)).astype(sim.dtype)
HS = ndi.convolve(
    cp.ones(sim.Nx_padded, dtype=sim.dtype), KERNEL, mode="constant", cval=0.0
)

# beta continuation
beta_of = lambda epoch: min(BETA0 * BETA_GROWTH ** (epoch // BETA_STEP), BETA_MAX)

name = "topopt_acoustic2D" + ("_amplify" if AMPLIFY else "")
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames" / name

# ------------------------------------ optimization -----------------------------------
n = int(active.size)
xval = np.full((n, 1), 0.5)
xold1, xold2 = xval.copy(), xval.copy()
low, upp = np.zeros((n, 1)), np.ones((n, 1))
xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
a0, a_mma, c_mma, d_mma = 1.0, np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

energy_air = target_energy(cp.zeros(sim.Nx_padded, dtype=sim.dtype))
cost_ref = None
cost_history = []

if args.animate:
    (ANIMATION_DIR / "optimization").mkdir(parents=True, exist_ok=True)
tic = time.time()
for epoch in range(EPOCHS):
    beta = beta_of(epoch)
    gamma, x_tilde = physical(xval, beta)

    cost, grad_gamma = compute_sensitivity_pml(sim, gamma, source, sensors, um, sponge)

    dpx = dprojection(x_tilde, beta, ETA) * design
    grad_obj = filter_adjoint(grad_gamma * dpx).ravel()[active].get()

    cost_ref = cost if cost_ref is None else cost_ref
    sign = -1.0 if AMPLIFY else 1.0
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
            gamma.get()[crop].T,
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
gamma_final, _ = physical(xval, beta_of(EPOCHS))
gamma_binary = (gamma_final > 0.5).astype(sim.dtype) * design

energy_design = target_energy(gamma_final)
energy_binary = target_energy(gamma_binary)
db = lambda e: 10 * math.log10(e / energy_air)
print(
    f"air baseline   {db(energy_air):.2f} dB\n"  # energy_air is reference -> 0 dB
    f"design         {db(energy_design):.2f} dB\n"
    f"binarized      {db(energy_binary):.2f} dB  "
    f"(volume fraction {float(gamma_binary.sum() / design_area):.3f})"
)

design = gamma_binary.get()[crop]

if args.animate:
    wave_dir = ANIMATION_DIR / "wavefield"
    wave_dir.mkdir(parents=True, exist_ok=True)
    frames = simulate_frames(gamma_binary, record_every=max(1, N // 200))
    scale = max(float(np.max(np.abs(f))) for f in frames)
    overlay = np.ma.masked_where(design < 0.5, design)
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
elif args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 100, RESOLUTION[1] / 100), dpi=150)
    ax.imshow(design.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RESULTS_DIR / f"{name}.pdf", transparent=True)
    plt.close()
else:
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    axes[0].semilogy(cost_history, "k")
    axes[1].imshow(design.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
