import math
import time

import cupy as cp
import numpy as np
import torch
from cuwave.scalar import AcousticWave
from cuwave.signals import sineburst
from cuwave.utils import interior_slice, point_source
from cuwave.wave import simulate, stable_dt
from helper import (
    SETTINGS,
    benchmark_parser,
    load_phase,
    load_setup,
    plot_field,
    save_solution,
)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = benchmark_parser(setup="burst")
MATERIAL = SETTINGS["materials"]["wave"]
parser.add_argument("--rho-min", type=float, default=MATERIAL["rho"][0])  # heavy and stiff:
parser.add_argument("--rho-max", type=float, default=MATERIAL["rho"][1])
parser.add_argument("--kappa-min", type=float, default=MATERIAL["kappa"][0])  # sound-hard holes
parser.add_argument("--kappa-max", type=float, default=MATERIAL["kappa"][1])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
PRECISION = "float32"
SPACE_ORDER = 4
THREADS = (4, 128)

# physics (gray value 0 takes the min values, which may exceed the max; 1 / rho and
# 1 / kappa interpolate)
DURATION = 0.7  # 1.2  # simulated time
SAFETY = 0.95  # fraction of the stable time step

# ------------------------------------- load data -------------------------------------
g = load_phase(args)
bc, sources = load_setup(args)  # edges reflect (homogeneous Neumann), bc unused

# --------------------------------------- setup ---------------------------------------
R = args.resolution
R1, R2 = g.shape
Nx, dx = (R1 + 3, R2 + 3), (1.0 / R,) * 2  # R + 1 interior nodes plus one ghost ring
speed = math.sqrt(max(args.kappa_min / args.rho_min, args.kappa_max / args.rho_max))
dt = SAFETY * stable_dt(dx, speed, SPACE_ORDER)
N = math.ceil(DURATION / dt)

sim = AcousticWave(
    Nx,
    dx,
    N,
    dt,
    THREADS,
    precision=PRECISION,
    space_order=SPACE_ORDER,
    rho1=args.rho_min,
    rho2=args.rho_max,
    kappa1=args.kappa_min,
    kappa2=args.kappa_max,
)

# phase on the nodes, edge values continued into the ghost ring and the padding
nodal = np.pad(g, 1, mode="edge")
nodal = np.maximum.reduce(
    [nodal[1:, 1:], nodal[:-1, 1:], nodal[1:, :-1], nodal[:-1, :-1]]
)
pad = [(1, p - n - 1) for p, n in zip(sim.Nx_padded, nodal.shape)]
indicator = cp.asarray(np.pad(nodal, pad, mode="edge"), dtype=sim.dtype)

# sources: points, or gaussians as weighted point sources on the nearby nodes, both
# scaled by the nodal phase so that nothing is injected into holes
t = np.arange(N) * dt
coords, signals = [], []
x, y = np.meshgrid(np.linspace(0.0, 1.0, R1 + 1), np.linspace(0.0, R2 / R, R2 + 1), indexing="ij")
x, y = x.ravel(), y.ravel()
phase = nodal.ravel()
for s in sources:
    burst = sineburst(t, s["amplitude"][0], s["frequency"], s["cycles"])
    position = (s["position"][0], s["position"][1] * R2 / R)  # fractions of each side
    if s["kind"] == "point":
        i, j = (int(round(c * n)) for c, n in zip(s["position"], g.shape))
        coords.append(np.array([position]))
        signals.append(burst[:, None] * nodal[i, j])
        continue
    r2 = (x - position[0]) ** 2 + (y - position[1]) ** 2
    near = r2 < (4 * s["width"]) ** 2
    weight = (
        np.exp(-0.5 * r2[near] / s["width"] ** 2) / (2 * np.pi * s["width"] ** 2) / R**2
    )
    weight *= phase[near]
    coords.append(np.stack([x[near], y[near]], axis=1))
    signals.append(burst[:, None] * weight[None])
source = point_source(sim, np.concatenate(coords), np.concatenate(signals, axis=1))

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
u = simulate(sim, source, indicator)
cp.cuda.Stream.null.synchronize()
print(f"elapsed time {time.time() - tic:.2f} s for {N} steps")

# ----------------------------------- postprocessing ----------------------------------
pressure = u[interior_slice(sim)].get()
if args.save:
    save_solution(args, "wave", pressure=pressure, time=N * dt, g=g)
limit = np.percentile(np.abs(pressure), 98)  # a source can dominate the maximum
shown = np.where(nodal > 0.5, pressure, 0.0)  # the pressure inside holes is meaningless
plot_field(shown, g, "seismic", "benchmark_wave", args.book, vmin=-limit, vmax=limit)
