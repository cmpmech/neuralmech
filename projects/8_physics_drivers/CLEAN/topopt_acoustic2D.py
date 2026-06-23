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
    build_materials,
    compile_kernels,
    define_excitation,
    define_homogeneous_Neumann_BC,
    define_step_method,
    setup_simulation,
    setup_source,
    simulate,
)
from solvers.wave_sensitivity import compute_sensitivity, compute_subtracted_kernels

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_acoustic2D"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# transient acoustic topology optimization (TATO, the acoustic black hole of
# Herrmann et al. 2026, Fig. 2): a sine-burst point source on the left wall sends a wave
# down an open central channel towards a target box on the right wall. Aluminium is
# distributed in the upper and lower blocks FLANKING the channel so the structure traps
# and diverts the wave out of the waveguide before it reaches the target -- the box is
# never directly walled off, the channel stays open

# geometry (metres): x is horizontal (axis 0), y is vertical (axis 1)
LENGTHS = (18.0, 9.0)
SOURCE_CENTER = (0.3, 4.5)  # point source on the left wall, mid-height
CHANNEL_HEIGHT = 2.0  # open waveguide channel through the middle
DESIGN_X0 = 6.0  # flanking design blocks start here (left third open)
TARGET_CENTER = (17.0, 4.5)  # target box at the channel mouth on the right wall
TARGET_SIZE = (2.0, 2.0)

# discretization (axis 1 is padded to a multiple of 32 by the solver)
RESOLUTION = (128, 64)
CFL = 0.4

# physics: air (gamma = 0) <-> aluminium (gamma = 1), linear in inverse density and
# inverse bulk modulus. kappa2 is the transient-friendly value (Herrmann et al. 2026)
RHO1, RHO2 = 1.204, 2643.0
KAPPA1, KAPPA2 = 1.419e5, 6.87e8
FREQUENCY = 70.0  # source centre frequency [Hz]
CYCLES = 3
AMPLITUDE = 1e3
T = 0.18  # simulated time window [s]

# optimization (MMA with a volume constraint, beta-continuation for a binary design)
EPOCHS = 160
K_FACTOR = 1e0  # superposition scaling for the adjoint sensitivity
VOLFRAC = 0.4  # material budget inside the design region
RMIN = 2.0  # density-filter radius [cells]
ETA = 0.5  # projection threshold
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
    # scatter active design variables into the grid, filter, project, pin to the region
    full = cp.zeros(sim.Nx_padded, dtype=sim.dtype).ravel()
    full[ACTIVE] = cp.asarray(xval.ravel(), dtype=sim.dtype)
    x_tilde = density_filter(full.reshape(sim.Nx_padded))
    return projection(x_tilde, beta, ETA) * DESIGN, x_tilde


def target_energy(gamma):
    _, um = simulate(sim, source, gamma, sensors=sensors)
    return float(np.prod(sim.dx)) * sim.dt * float(cp.sum(um**2))


def simulate_frames(gamma, record_every):
    # forward wavefield, snapshots every record_every steps (for the animation)
    mat = build_materials(sim, gamma)
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
            frames.append(u1[1 : Nx[0] - 1, 1 : Nx[1] - 1].get())
    return frames


# --------------------------------------- setup ---------------------------------------
Nx = RESOLUTION
dx = tuple(LENGTHS[d] / (Nx[d] - 3) for d in range(2))
to_index = lambda coord: tuple(int(round(coord[d] / dx[d])) for d in range(2))

# stable time step from the fastest wave speed c = sqrt(kappa / rho) over the design
g_scan = cp.linspace(0, 1, 21)
rho_inv = 1 / RHO1 + g_scan * (1 / RHO2 - 1 / RHO1)
kappa_inv = 1 / KAPPA1 + g_scan * (1 / KAPPA2 - 1 / KAPPA1)
wavespeed = float(cp.max(cp.sqrt(rho_inv / kappa_inv)))
dt = CFL * min(dx) / wavespeed / math.sqrt(2)
N = math.ceil(T / dt)

sim = setup_simulation(
    Nx,
    dx,
    N,
    dt,
    wavespeed,
    RHO1,
    (4, 64),
    formulation="acoustic",
    precision="float32",
    rho1=RHO1,
    rho2=RHO2,
    kappa1=KAPPA1,
    kappa2=KAPPA2,
)

# sine-burst point source
t_arr = np.linspace(0, (N - 1) * dt, N)
signal_np = sineburst(t_arr, AMPLITUDE, FREQUENCY, CYCLES) / np.prod(dx)
signal = cp.asarray(signal_np[:, None], dtype=sim.dtype)
src_i, src_j = to_index(SOURCE_CENTER)
source = setup_source(cp.array([[src_i], [src_j]], dtype=cp.int32), signal)

# target box Omega_s: every grid cell inside is a "sensor" we drive towards zero pressure
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

# design region: upper and lower blocks flanking the open central channel that carries
# the wave from the source to the target (the channel and the left third stay air)
DESIGN = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
x0 = to_index((DESIGN_X0, 0))[0]
ch_lo = to_index((0, LENGTHS[1] / 2 - CHANNEL_HEIGHT / 2))[1]
ch_hi = to_index((0, LENGTHS[1] / 2 + CHANNEL_HEIGHT / 2))[1]
DESIGN[x0 : Nx[0] - 1, 1:ch_lo] = 1.0  # lower block
DESIGN[x0 : Nx[0] - 1, ch_hi : Nx[1] - 1] = 1.0  # upper block
DESIGN_AREA = float(DESIGN.sum())
ACTIVE = cp.where(DESIGN.ravel() > 0)[0]  # flat indices of the design variables

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

# ------------------------------------ optimization -----------------------------------
n = int(ACTIVE.size)
xval = np.full((n, 1), VOLFRAC)
xold1, xold2 = xval.copy(), xval.copy()
low, upp = np.zeros((n, 1)), np.ones((n, 1))
xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
a0, a_mma, c_mma, d_mma = 1.0, np.zeros((1, 1)), 1e3 * np.ones((1, 1)), np.zeros((1, 1))

energy_air = target_energy(cp.zeros(sim.Nx_padded, dtype=sim.dtype))
cost_ref = None
cost_history = []

tic = time.time()
if args.animate:
    (ANIMATION_DIR / "optimization").mkdir(parents=True, exist_ok=True)
for epoch in range(EPOCHS):
    beta = beta_of(epoch)
    gamma, x_tilde = physical(xval, beta)

    sk = cp.zeros(sim.Nx_padded, dtype=sim.dtype)
    cost, sk = compute_subtracted_kernels(sk, sim, source, gamma, um, sensors, K_FACTOR)
    grad_gamma = compute_sensitivity(sim, sk, K_FACTOR, 1)

    # chain rule back through projection and filter for objective and volume constraint
    dpx = dprojection(x_tilde, beta, ETA) * DESIGN
    grad_obj = filter_adjoint(grad_gamma * dpx).ravel()[ACTIVE].get()
    grad_vol = filter_adjoint(dpx / DESIGN_AREA).ravel()[ACTIVE].get()

    cost_ref = cost if cost_ref is None else cost_ref
    volume = float(gamma.sum()) / DESIGN_AREA
    f0val = cost / cost_ref
    df0dx = (grad_obj / cost_ref).reshape(n, 1)
    fval = np.array([[volume / VOLFRAC - 1.0]])
    dfdx = (grad_vol / VOLFRAC).reshape(1, n)

    xmma, _, _, _, _, _, _, _, _, low, upp = mmapy.mmasub(
        1,
        n,
        epoch + 1,
        xval,
        xmin,
        xmax,
        xold1,
        xold2,
        f0val,
        df0dx,
        fval,
        dfdx,
        low,
        upp,
        a0,
        a_mma,
        c_mma,
        d_mma,
        move=MMA_MOVE,
    )
    xold2, xold1 = xold1, xval
    xval = xmma

    cost_history.append(cost / energy_air)
    print(
        f"epoch {epoch}: reduction {10 * math.log10(cost / energy_air):.2f} dB  "
        f"vol {volume:.3f}  beta {beta:.0f}"
    )

    if args.animate:
        fig, ax = plt.subplots(figsize=(Nx[0] / 100, Nx[1] / 100), dpi=150)
        ax.imshow(
            gamma.get()[1 : Nx[0] - 1, 1 : Nx[1] - 1].T,
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
gamma_binary = (gamma_final > 0.5).astype(sim.dtype) * DESIGN

energy_design = target_energy(gamma_final)
energy_binary = target_energy(gamma_binary)
db = lambda e: 10 * math.log10(e / energy_air)
print(
    f"air baseline   0.00 dB\n"
    f"design         {db(energy_design):.2f} dB\n"
    f"binarized      {db(energy_binary):.2f} dB  "
    f"(volume fraction {float(gamma_binary.sum() / DESIGN_AREA):.3f})"
)

design = gamma_binary.get()[1 : Nx[0] - 1, 1 : Nx[1] - 1]

if args.animate:
    # second animation: the transient wavefield through the optimized binary structure,
    # with the structure overlaid (cf. the field plots in topopt_helmholtz.py)
    wave_dir = ANIMATION_DIR / "wavefield"
    wave_dir.mkdir(parents=True, exist_ok=True)
    frames = simulate_frames(gamma_binary, record_every=max(1, N // 200))
    scale = max(float(np.max(np.abs(f))) for f in frames)
    overlay = np.ma.masked_where(design < 0.5, design)
    for f, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(Nx[0] / 100, Nx[1] / 100), dpi=150)
        ax.imshow(frame.T, origin="lower", cmap="seismic", vmin=-scale, vmax=scale)
        ax.imshow(overlay.T, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.85)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(wave_dir / f"frame_{f:04d}.jpg")
        plt.close()
elif args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(Nx[0] / 100, Nx[1] / 100), dpi=150)
    ax.imshow(design.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RESULTS_DIR / "topopt_acoustic2D.png")
    plt.close()
else:
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    axes[0].semilogy(cost_history, "k")
    axes[1].imshow(design.T, origin="lower", cmap="binary", vmin=0, vmax=1)
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
