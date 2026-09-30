import math
import time

import cmasher as cmr
import cupy as cp
import numpy as np
import torch
from scipy import ndimage
from cufluid.lbm.boundary import Outflow, Wall
from cufluid.lbm.lbm import LatticeBoltzmann, moments, simulate

from helper import EDGES, SETTINGS, benchmark_parser, load_phase, load_setup, plot_field, save_solution

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = benchmark_parser(setup="channel")
parser.add_argument("--reynolds", type=float, default=SETTINGS["materials"]["fluid"]["reynolds"])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# implementation
PRECISION = "float32"
THREADS = (4, 128)

# discretization (lattice units, one node per pixel)
VELOCITY = 0.002  # inflow; density varies with its square, keeping it within 3 %
DURATION = 3  # flow-through times of the domain

# ------------------------------------- load data -------------------------------------
g = load_phase(args)
bc, sources = load_setup(args)
threshold = SETTINGS["geometry"]["threshold"] if args.threshold is None else args.threshold
solid = g < threshold  # the weak phase is solid, the fluid flows through the matrix


# --------------------------------------- helper --------------------------------------
def face(edge):
    """cufluid condition of a whole edge: a (moving) wall when prescribed, else outflow."""
    e, nodes = EDGES.index(edge), bc.edge_nodes(edge)  # without the padding
    dirichlet = bc.dirichlet[e, :nodes].numpy()
    if not dirichlet.any():
        return Outflow()
    if not dirichlet.all():
        raise ValueError(f"cufluid needs one condition per face, {edge} is mixed")
    velocity = VELOCITY * bc.u[e, :nodes].numpy().mean(axis=0)
    return Wall(velocity=tuple(velocity)) if np.abs(velocity).max() > 0 else Wall()


# fluid cut off from every outflow only accumulates the inflowing mass, so it is solid too
outflow = [e for e in EDGES if not bc.dirichlet[EDGES.index(e)].numpy().any()]
labels, _ = ndimage.label(~solid, structure=np.ones((3, 3)))  # D2Q9 streams diagonally
edges = {
    "left": labels[0],
    "right": labels[-1],
    "bottom": labels[:, 0],
    "top": labels[:, -1],
}
open_labels = np.unique(np.concatenate([edges[e] for e in outflow]))
solid |= ~np.isin(labels, open_labels[open_labels > 0])

# --------------------------------------- setup ---------------------------------------
R1, R2 = g.shape  # the channel runs along the long side x_1
Nx = (R1 + 2, R2 + 2)  # plus one ghost layer
viscosity = VELOCITY * R1 / args.reynolds  # with the long side, as stable as the square
N = math.ceil(DURATION * R1 / VELOCITY)

mask = cp.zeros(Nx, dtype=cp.bool_)
mask[1:-1, 1:-1] = cp.asarray(solid)
sim = LatticeBoltzmann(
    Nx,
    N,
    THREADS,
    precision=PRECISION,
    viscosity=viscosity,
    boundary=((face("left"), face("right")), (face("bottom"), face("top"))),
    solid=mask,
)

# --------------------------------------- solve ---------------------------------------
cp.cuda.Stream.null.synchronize()
tic = time.time()
f = simulate(sim, u0=(0.0, 0.0))
cp.cuda.Stream.null.synchronize()
print(f"elapsed time {time.time() - tic:.2f} s for {N} steps")

# ----------------------------------- postprocessing ----------------------------------
rho, u = moments(sim, f)
velocity = np.nan_to_num(u.get()) / VELOCITY  # per pixel, zero in the solid
vorticity = np.gradient(velocity[1], axis=0) - np.gradient(velocity[0], axis=1)

if args.save:
    save_solution(args, "fluid", velocity=velocity, pressure=rho.get() / 3, g=g)
limit = np.percentile(np.abs(vorticity), 95)
plot_field(
    vorticity,
    g,
    cmr.guppy,
    "benchmark_fluid",
    args.book,
    vmin=-limit,
    vmax=limit,
    solid=0.8,
)
