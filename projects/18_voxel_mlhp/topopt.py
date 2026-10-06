import argparse
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from solvers.optimization import DensityFilter, simp
from voxel import VoxelMultigrid

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--resolution", type=int, default=160)  # voxels per unit length
# (p, s) = (2, 4) matches the voxel reference here, under low-load regions only (2, 3)
parser.add_argument("--degree", type=int, default=2)
parser.add_argument("--sub", type=int, default=4)  # voxels per element and direction
parser.add_argument("--floor", type=float, default=1e-3)  # element-wise alpha, 0 off
parser.add_argument("--filter", default="density", choices=["density", "sensitivity"])
parser.add_argument("--cycle", choices=["V", "W"])  # default W in 3D, V in 2D
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

# 2D: half MBB beam (left edge is the symmetry plane), 3D: cantilever with a line
# load along the bottom edge of its free end
# geometry
LENGTHS = [4.0, 1.0] if D == 2 else [2.0, 1.0, 1.0]

# physics
VOLFRAC = 0.5 if D == 2 else 0.15
PENAL = 3.0
RMIN = 2  # filter radius in voxels
E0, EMIN = 1.0, 1e-9
LOAD = -1.0

# optimization (optimality criterion)
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 300
CHANGE_TOL = 0.01

# solver
CG_TOL = 1e-5  # relative residual, sensitivities to ~1e-6
CYCLE = args.cycle or ("W" if D == 3 else "V")  # W pays where coarse meshes are cheap

# postprocessing
THRESHOLD = 0.5

# ---------------------------------------- helper -------------------------------------
# fused kernels, no design-sized temporaries: these set the memory peak at large grids
OC_STEP = "min(min((T)1, x + move), max(max((T)0, x - move), base * scale))"
oc_update = cp.ElementwiseKernel(
    "T x, T base, T scale, T move", "T x_new", f"x_new = {OC_STEP}", "oc_update"
)
oc_volume = cp.ReductionKernel(
    "T x, T base, T scale, T move, T volume",
    "float64 total",
    f"(double)({OC_STEP}) * volume",
    "a + b",
    "total = a",
    "0",
    "oc_volume",
)
oc_base = cp.ElementwiseKernel(
    "T x, T volume, T damping",
    "T dc",
    "dc = x * pow(max((T)0, -dc) / volume, damping)",
    "oc_base",
)
simp_chain = cp.ElementwiseKernel(  # g *= -dsimp(rho)
    "T rho, T penal, T scale",
    "T g",
    "g *= -penal * pow(rho, penal - 1) * scale",
    "simp_chain",
)
max_change = cp.ReductionKernel(
    "T a, T b", "T y", "abs(a - b)", "max(a, b)", "y = a", "0", "max_change"
)


def thin_voxels(solid):
    """fraction of voxels that differ from both neighbours along an axis."""
    thin = cp.zeros(solid.shape, dtype=bool)
    for axis in range(solid.ndim):
        inner = [slice(None)] * solid.ndim
        inner[axis] = slice(1, -1)
        left, right = list(inner), list(inner)
        left[axis], right[axis] = slice(None, -2), slice(2, None)
        centre = solid[tuple(inner)]
        thin[tuple(inner)] |= (centre != solid[tuple(left)]) & (
            centre != solid[tuple(right)]
        )
    return float(thin.mean())


def create_solver(degree, sub):
    nodes = [n // sub * degree + 1 for n in nvoxels]
    fixed = np.zeros((D, *nodes), dtype=bool)
    if D == 2:
        fixed[0, 0, :] = True  # symmetry
        fixed[1, -1, 0] = True  # roller
    else:
        fixed[:, 0] = True
    solver = VoxelMultigrid(
        nvoxels,
        LENGTHS,
        "elasticity",
        fixed,
        degree,
        sub,
        floor=args.floor,
        w_cycle=CYCLE == "W",
    )
    if D == 2:
        force = cp.zeros((2, *nodes))
        force[1, 0, -1] = LOAD
        return solver, force.ravel()
    # line load along the bottom edge of the free end, consistent along z
    face = solver.face_load(0, 1, [0.0, LOAD, 0.0]).reshape(D, *nodes)
    force = cp.zeros_like(face)
    force[1, -1, 0] = face[1, -1, 0] / face[1, -1, 0].sum() * LOAD * LENGTHS[2]
    return solver, force.ravel()


# ---------------------------------------- setup --------------------------------------
nvoxels = [int(args.resolution * length) for length in LENGTHS]
solver, force = create_solver(args.degree, args.sub)
print(f"voxels {nvoxels}, dofs {solver.ndof}, levels {len(solver.levels)}")
density_filter = DensityFilter(RMIN, nvoxels, xp=cp, dtype=cp.float32)

# ------------------------------------ optimization -----------------------------------
x = cp.full(nvoxels, VOLFRAC, dtype=cp.float32)
dv = density_filter.adjoint(cp.ones(nvoxels, cp.float32))  # d mean(rho) / dx * N
u = None
cg_history = []

tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    rho = density_filter(x) if args.filter == "density" else x
    solver.update(simp(rho, PENAL, EMIN, E0))
    u, cg_iters = solver.solve(force, x0=u, rtol=CG_TOL)
    compliance = float(force @ u)

    dc = solver.energy_gradient(u)
    simp_chain(rho, cp.float32(PENAL), cp.float32(E0 - EMIN), dc)
    if args.filter == "density":
        dc = density_filter.adjoint(dc)
    else:
        dc = density_filter.sensitivity(x, dc)

    # optimality criterion update with bisection on the log volume multiplier
    volume = dv if args.filter == "density" else cp.ones_like(x)
    base = oc_base(x, volume, cp.float32(DAMPING), dc)
    move = cp.float32(MOVE)
    l1, l2 = 1e-9, 1e9
    while l2 / l1 > 1.0 + 2e-4:
        lmid = (l1 * l2) ** 0.5
        scale = cp.float32(lmid**-DAMPING)
        if float(oc_volume(x, base, scale, move, volume)) / x.size > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid
    x_new = oc_update(x, base, scale, move, base)

    change = float(max_change(x_new, x))
    x = x_new
    cg_history.append(cg_iters)
    pbar.set_postfix(
        {"c": f"{compliance:.4e}", "change": f"{change:.2e}", "cg": cg_iters}
    )
    if change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it + 1} iter, "
    f"time per iter {(toc - tic) / (it + 1):.2e} s, "
    f"cg iterations mean {np.mean(cg_history):.1f} max {max(cg_history)}, "
    f"gpu memory {cp.get_default_memory_pool().total_bytes() / np.prod(nvoxels):.0f} "
    f"B/voxel (pool peak)"
)

# ----------------------------------- postprocessing ----------------------------------
# the thresholded design on the voxel reference (1, 1) exposes artificially stiff
# patterns of the coarser analysis
rho = density_filter(x) if args.filter == "density" else x
solid = rho > THRESHOLD
del solver
reference, force = create_solver(1, 1)
reference.update(simp(solid.astype(cp.float32), PENAL, EMIN, E0))
u_ref, _ = reference.solve(force, rtol=CG_TOL)
print(
    f"compliance {compliance:.4e}, thresholded on voxels {float(force @ u_ref):.4e}, "
    f"thin voxels {thin_voxels(solid):.2e}"
)

image = rho if D == 2 else rho.mean(axis=2)  # 3D: averaged through the depth
fig, ax = plt.subplots(figsize=(12.8, 12.8 * LENGTHS[1] / LENGTHS[0]), dpi=150)
ax.imshow(image.get().T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
