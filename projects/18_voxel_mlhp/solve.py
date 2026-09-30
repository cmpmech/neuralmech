import argparse
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import materials
from voxel import VoxelMultigrid

parser = argparse.ArgumentParser()
parser.add_argument(
    "--physics", default="elasticity", choices=["elasticity", "poisson"]
)
parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
parser.add_argument(
    "--material", default="gyroid", choices=["gyroid", "inclusions", "random"]
)
parser.add_argument("--resolution", type=int, default=128)  # voxels per unit length
parser.add_argument("--degree", type=int, default=2)
parser.add_argument("--sub", type=int, default=4)  # voxels per element and direction
parser.add_argument("--floor", type=float, default=1e-3)  # element-wise alpha, 0 off
parser.add_argument("--contrast", type=float, default=None)
parser.add_argument("--smoothing", type=float, default=0.0)  # gaussian width
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

# geometry
LENGTHS = [2.0, 1.0, 1.0][:D]

# physics
CONTRAST = {"gyroid": 1e-6, "inclusions": 1e3, "random": 1e3}[args.material]
CONTRAST = CONTRAST if args.contrast is None else args.contrast
LOAD = [0.0, -1.0, 0.0][:D] if args.physics == "elasticity" else [1.0]

# solver
CG_TOL = 1e-8

# ------------------------------------- create data -----------------------------------
nvoxels = [int(args.resolution * length) for length in LENGTHS]
if args.material == "gyroid":
    indicator = materials.gyroid(nvoxels, LENGTHS)
elif args.material == "inclusions":
    indicator = materials.inclusions(nvoxels, LENGTHS)
else:
    indicator = materials.random_field(nvoxels, LENGTHS)
if args.smoothing > 0.0:
    indicator = materials.smooth(indicator, args.smoothing, nvoxels, LENGTHS)
if args.material == "random":  # log-normal, +-2 standard deviations span CONTRAST
    coeff = cp.exp(0.25 * np.log(CONTRAST) * indicator).astype(cp.float32)
else:
    coeff = materials.two_phase(indicator, CONTRAST)

# ---------------------------------------- solve --------------------------------------
pool = cp.get_default_memory_pool()
nodes = [n // args.sub * args.degree + 1 for n in nvoxels]
fixed = np.zeros((len(LOAD), *nodes), dtype=bool)
fixed[:, 0] = True
solver = VoxelMultigrid(
    nvoxels, LENGTHS, args.physics, fixed, args.degree, args.sub, floor=args.floor
)
force = solver.face_load(0, 1, LOAD)
solver.update(coeff)
solver.solve(force, rtol=CG_TOL)  # compiles the kernels and captures the graph

cp.cuda.Device().synchronize()
tic = time.time()
solver.update(coeff)
setup = time.time() - tic
tic = time.time()
u, iters = solver.solve(force, rtol=CG_TOL)
solve = time.time() - tic
print(
    f"{np.prod(nvoxels)} voxels, dofs {solver.ndof} cg {iters} setup {setup:.2f} s "
    f"solve {solve:.2f} s {pool.used_bytes() / coeff.size:.1f} B/voxel "
    f"compliance {float(force @ u):.6e}"
)

# ----------------------------------- postprocessing ----------------------------------
# material and |u| at the voxel corners, mid-plane slice in 3D
field = np.linalg.norm(solver.voxel_nodes(u).get(), axis=0)
images = [(coeff.get(), "cividis", LogNorm()), (field, "turbo", None)]
if D == 3:
    images = [(image[..., image.shape[2] // 2], *rest) for image, *rest in images]
fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=100)
for ax, (image, cmap, norm) in zip(axes, images):
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm)
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
