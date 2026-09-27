import argparse
import time

import cupy as cp
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.ndimage

from helper import VoxelMultigrid

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

# geometry
LENGTHS = [2.0, 1.0, 1.0][:D]
CELLS = 4  # gyroid unit cells along x
THICKNESS = 0.4  # gyroid sheet half-thickness in level-set units
PLATE = 0.05  # solid end plates as a fraction of the length

# discretization
RESOLUTIONS = (
    [32, 64, 128] if D == 3 else [128, 256, 512, 1024]
)  # voxels per unit length
DEGREE = 1
SUB_VOXELS = 2  # voxels per element edge

# physics
E_SOLID, ALPHA, NU = 1.0, 1e-6, 0.3
TRACTION = -1.0

# solver
CG_TOL = 1e-8


# ---------------------------------------- helper -------------------------------------
def gyroid(nvoxels):
    axes = [(np.arange(n) + 0.5) * length / n for n, length in zip(nvoxels, LENGTHS)]
    x, y, z = np.meshgrid(*axes, *([[0.25]] if D == 2 else []), indexing="ij")
    k = 2.0 * np.pi * CELLS / LENGTHS[0]
    level_set = (
        np.sin(k * x) * np.cos(k * y)
        + np.sin(k * y) * np.cos(k * z)
        + np.sin(k * z) * np.cos(k * x)
    )
    plates = (x < PLATE * LENGTHS[0]) | (x > (1.0 - PLATE) * LENGTHS[0])
    solid = ((np.abs(level_set) < THICKNESS) | plates).reshape(nvoxels)
    # floating islands only add near-singular rigid body modes, keep the main body
    labels, _ = scipy.ndimage.label(solid)
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def clamped(basis):  # all displacement components on the x-min face
    bc = [
        mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [0], ifield=f)
        for f in range(D)
    ]
    return np.array(mlhp.combineDirichletDofs(bc)[0])


# ---------------------------------------- setup --------------------------------------
if D == 2:
    material = mlhp.planeStressMaterial(
        mlhp.scalarField(D, 1.0), mlhp.scalarField(D, NU)
    )
else:
    material = mlhp.isotropicElasticMaterial(
        mlhp.scalarField(D, 1.0), mlhp.scalarField(D, NU)
    )
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(D), material, mlhp.vectorField(D, [0.0] * D)
)
load = mlhp.neumannIntegrand(mlhp.vectorField(D, [0.0, TRACTION, 0.0][:D]))

# ---------------------------------------- solve --------------------------------------
for resolution in RESOLUTIONS:
    nvoxels = [int(resolution * length) for length in LENGTHS]
    solid = gyroid(nvoxels)
    coeff = cp.asarray(np.where(solid, E_SOLID, ALPHA * E_SOLID))

    tic = time.time()
    solver = VoxelMultigrid(integrand, nvoxels, LENGTHS, DEGREE, SUB_VOXELS, D, clamped)
    nel = [n // SUB_VOXELS for n in nvoxels]
    mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nel, lengths=LENGTHS))
    force = mlhp.DoubleVector(solver.ndof)
    mlhp.integrateOnSurface(
        solver.basis, load, [force], mlhp.quadratureOnMeshFaces(mesh, [1])
    )
    force = cp.asarray(np.array(force.array))
    build = time.time() - tic

    solver.update(coeff)
    solver.solve(force, rtol=CG_TOL)
    tic = time.time()
    solver.update(coeff)
    setup = time.time() - tic
    u, iters = solver.solve(force, rtol=CG_TOL)
    solve = time.time() - tic - setup

    print(
        f"voxels {np.prod(nvoxels):9d}  dofs {solver.ndof:9d}  "
        f"levels {len(solver.levels)}  cg iterations {iters:3d}  "
        f"build {build:6.2f} s  setup {1e3 * setup:7.1f} ms  "
        f"solve {1e3 * solve:7.1f} ms  "
        f"gpu memory {cp.get_default_memory_pool().total_bytes() / 1e9:.2f} GB"
    )
    del solver
    cp.get_default_memory_pool().free_all_blocks()

# ----------------------------------- postprocessing ----------------------------------
# element-averaged displacement magnitude, mid-plane slice in 3D
fine_efts = np.array(
    mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=D).locationMaps()
)
u_e = u.get()[fine_efts].reshape(len(fine_efts), D, -1).mean(axis=2)
magnitude = np.linalg.norm(u_e, axis=1).reshape(nel)
solid_e = solid.reshape([n for pair in zip(nel, [SUB_VOXELS] * D) for n in pair]).mean(
    axis=tuple(range(1, 2 * D, 2))
)
if D == 3:
    magnitude, solid_e = magnitude[:, :, nel[2] // 2], solid_e[:, :, nel[2] // 2]

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.imshow(
    np.ma.masked_where(solid_e.T < 0.5, magnitude.T), origin="lower", cmap="turbo"
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
