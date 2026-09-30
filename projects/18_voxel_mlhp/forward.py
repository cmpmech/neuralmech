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
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument(
    "--material", default="gyroid", choices=["gyroid", "inclusions", "random"]
)
parser.add_argument("--contrast", type=float, default=None)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

# geometry
LENGTHS = [2.0, 1.0, 1.0][:D]
SMOOTHING = 0.0  # standard deviation of a Gaussian smoothing of the material

# discretization
RESOLUTIONS = [256, 512, 1024, 2048] if D == 2 else [32, 64, 128]  # voxels per length
SETTINGS = [(1, 2), (2, 2), (2, 4), (3, 4), (3, 8)]  # (p, s), reference is (1, 1)

# physics
CONTRAST = {"gyroid": 1e-6, "inclusions": 1e3, "random": 1e3}[args.material]
CONTRAST = CONTRAST if args.contrast is None else args.contrast
LOAD = [0.0, -1.0, 0.0][:D] if args.physics == "elasticity" else [1.0]

# solver
CG_TOL = 1e-8


# ---------------------------------------- helper -------------------------------------
def material(nvoxels):
    if args.material == "gyroid":
        indicator = materials.gyroid(nvoxels, LENGTHS)
    elif args.material == "inclusions":
        indicator = materials.inclusions(nvoxels, LENGTHS)
    else:
        indicator = materials.random_field(nvoxels, LENGTHS)
    if SMOOTHING > 0.0:
        indicator = materials.smooth(indicator, SMOOTHING, nvoxels, LENGTHS)
    if args.material == "random":  # log-normal, +-2 standard deviations span CONTRAST
        return cp.exp(0.25 * np.log(CONTRAST) * indicator).astype(cp.float32)
    return materials.two_phase(indicator, CONTRAST)


def run(coeff, degree, sub):
    pool = cp.get_default_memory_pool()
    before = pool.used_bytes()
    nvoxels = coeff.shape
    nodes = [n // sub * degree + 1 for n in nvoxels]
    fixed = np.zeros((len(LOAD), *nodes), dtype=bool)
    fixed[:, 0] = True
    solver = VoxelMultigrid(nvoxels, LENGTHS, args.physics, fixed, degree, sub)
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
    memory = pool.used_bytes() - before
    return solver, force, u, iters, setup, solve, memory / coeff.size


# ---------------------------------------- solve --------------------------------------
for resolution in RESOLUTIONS:
    nvoxels = [int(resolution * length) for length in LENGTHS]
    coeff = material(nvoxels)
    reference, force_ref, u_ref, *stats = run(coeff, 1, 1)
    energy_ref = float(force_ref @ u_ref)
    print(
        f"{np.prod(nvoxels)} voxels, reference: dofs {reference.ndof} cg {stats[0]} "
        f"setup {stats[1]:.2f} s solve {stats[2]:.2f} s {stats[3]:.0f} B/voxel"
    )
    for degree, sub in SETTINGS:
        solver, force, u, iters, setup, solve, memory = run(coeff, degree, sub)
        energy = float(force @ u)
        e = solver.voxel_nodes(u).ravel() - u_ref
        error = float(e @ reference.apply(e)) ** 0.5 / energy_ref**0.5
        print(
            f"  p {degree} s {sub}: dofs {solver.ndof:9d} cg {iters:4d} "
            f"setup {setup:.2f} s solve {solve:.2f} s {memory:6.1f} B/voxel "
            f"energy error {(energy - energy_ref) / energy_ref:+.2e} "
            f"energy norm error {error:.2e}"
        )
        field = solver.voxel_nodes(u).get()
        del solver, u
    field_ref = u_ref.get().reshape(field.shape)
    del reference, u_ref

# ----------------------------------- postprocessing ----------------------------------
# material, reference |u| and the error of the last (p, s), mid-plane slice in 3D
images = [
    (coeff.get(), "cividis"),
    (np.linalg.norm(field_ref, axis=0), "turbo"),
    (np.linalg.norm(field - field_ref, axis=0), "hot_r"),
]
if D == 3:
    images = [(image[..., image.shape[2] // 2], cmap) for image, cmap in images]
fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=100)
for ax, (image, cmap) in zip(axes, images):
    norm = None if cmap == "turbo" else LogNorm()
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm)
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
