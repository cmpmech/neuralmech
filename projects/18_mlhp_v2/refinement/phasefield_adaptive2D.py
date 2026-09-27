import argparse
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

from helper import Multigrid, QuadTree, UnitSquare, isotropic_tensor

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--load", default="tension", choices=["tension", "shear"])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# single edge notched tension or shear (Miehe et al. 2010), units kN and mm
# geometry
LENGTHS = [1.0, 1.0]
NOTCH = 0.5  # notch length from the left edge at mid height, two voxels thick

# discretization
RESOLUTION = args.resolution  # finest elements per edge, a power of two
COARSE = 64  # elements per edge away from the crack
REFINE_D = 0.1  # phase field above which an element gets refined surroundings
BUFFER = 0.05  # radius refined around the notch and such elements

# physics
LAMBDA, MU = 121.15, 80.77  # plane strain
GC = 2.7e-3  # fracture toughness
ELL = 0.0075  # phase-field length scale
RESIDUAL = 1e-7  # stiffness left in fully broken material
ALPHA = 1e-6  # stiffness of the notch elements

# loading
LOAD = args.load
STAGES = {  # (end displacement, increment) of the top face
    "tension": [(0.005, 1e-4), (0.0065, 1e-6)],
    "shear": [(0.02, 1e-4)],
}[LOAD]

# solver
CG_TOL = 1e-6  # relative residual, per staggered iteration
SMOOTHING = (2,)  # Chebyshev degree per level, fine to coarse
STAGGER_TOL = 1e-4  # max phase-field change between staggered iterations
MAX_STAGGER = 500

# ---------------------------------------- setup --------------------------------------
depth = int(np.log2(RESOLUTION))
coarse = int(np.log2(COARSE))
square = UnitSquare()

# displacement: g K + (1 - g) [tr < 0] K_mean, with the element mean dilatation
D = 2
grad = square.gradient(D)
b_mean = square.mean(grad[:, 0] + grad[:, 3])
K_full = square.integrate(grad, isotropic_tensor(D, LAMBDA, MU))
K_mean = (LAMBDA + 2.0 * MU / 3.0) * np.outer(b_mean, b_mean)

# phase field: Gc (M / l + l L) + 2 H M_mean, all masses lumped
value = square.value(1)
m_mean = square.mean(value[:, 0])
M = np.diag(square.integrate(value).sum(axis=1))
M_mean = np.diag(np.outer(m_mean, m_mean).sum(axis=1))
L = square.integrate(square.gradient(1))
b_mean, m_mean = cp.asarray(b_mean), cp.asarray(m_mean)


def supports(xy):  # bottom and top faces clamped, sides guided vertically in shear
    clamped = (xy[:, 1] == 0.0) | (xy[:, 1] == LENGTHS[1])
    guided = (xy[:, 0] == 0.0) | (xy[:, 0] == LENGTHS[0])
    return np.stack([clamped, clamped | (guided & (LOAD == "shear"))])


def refine(tree, hot):  # finest level within BUFFER of the notch and of hot centers
    near = cKDTree(hot) if len(hot) else None

    def target(x, y):
        fine = (x < NOTCH + BUFFER) & (np.abs(y - 0.5 * LENGTHS[1]) < BUFFER)
        if near is not None:
            fine |= np.isfinite(
                near.query(np.stack([x, y], 1), distance_upper_bound=BUFFER)[0]
            )
        return np.where(fine, depth, coarse)

    return tree.refine_to(target)


def build(tree):  # everything that depends on the mesh
    solver_u = Multigrid(tree, [(K_full, 0), (K_mean, 0)], D, supports, SMOOTHING)
    solver_d = Multigrid(
        tree,
        [(M, 2), (L, 0), (M_mean, 2)],
        1,
        lambda xy: np.zeros((1, len(xy)), bool),
        SMOOTHING,
    )
    space = solver_u.space
    on_top = np.flatnonzero(space.xy[:, 1] == LENGTHS[1])
    top = cp.asarray(on_top + (1 if LOAD == "tension" else 0) * space.nfree)
    x, y = tree.centers.T
    notch = (x < NOTCH) & (np.abs(y - 0.5 * LENGTHS[1]) < LENGTHS[1] / RESOLUTION)
    stiffness = cp.asarray(np.where(notch, ALPHA, 1.0))
    return solver_u, solver_d, top, notch, stiffness, cp.asarray(tree.h**2)


tic = time.time()
tree = refine(QuadTree([1, 1], LENGTHS, depth), np.zeros((0, 2)))
solver_u, solver_d, top, notch, stiffness, h2 = build(tree)
print(f"elements {len(tree.level)}, setup {time.time() - tic:.2f} s")

# ------------------------------------- simulation ------------------------------------
loads = np.concatenate(
    [
        np.arange(start, end, inc) + inc
        for (start, _), (end, inc) in zip([(0.0, 0)] + STAGES, STAGES)
    ]
)
u = cp.zeros(solver_u.ndof)
w = None
d = cp.zeros(solver_d.ndof)
history = cp.zeros(len(tree.level))
reactions, staggers, cg_u, cg_d, remesh_time = [], [], [], [], 0.0

tic = time.time()
for step, load in enumerate(loads):
    u_dirichlet = cp.zeros(solver_u.ndof)
    u_dirichlet[top] = load
    w = u * (load / loads[step - 1]) - u_dirichlet if step else None  # linear predictor
    for stagger in range(MAX_STAGGER):
        g = (
            solver_d.element_values((1.0 - cp.clip(d, 0.0, 1.0)) ** 2, m_mean)
            + RESIDUAL
        )
        compressed = solver_u.element_values(u, b_mean) < 0.0
        solver_u.update([stiffness * g, stiffness * (1.0 - g) * compressed], False)
        lift = -solver_u.apply_full(u_dirichlet)
        w, it = solver_u.solve(lift, x0=w, rtol=CG_TOL)
        cg_u.append(it)
        u = u_dirichlet + w
        # the tension/compression switch settles with the staggered iteration
        switched = bool(
            (compressed != (solver_u.element_values(u, b_mean) < 0.0)).any()
        )
        energy_full, energy_mean = solver_u.element_energy(u)
        psi_plus = 0.5 * stiffness * (energy_full - compressed * energy_mean) / h2
        driving = cp.maximum(history, psi_plus)
        nel = len(tree.level)
        solver_d.update(
            [cp.full(nel, GC / ELL), cp.full(nel, GC * ELL), 2.0 * driving], False
        )
        rhs = solver_d.load(2.0 * driving, m_mean, 2)
        d_new, it = solver_d.solve(rhs, x0=d, rtol=CG_TOL)
        cg_d.append(it)
        change = float(cp.abs(d_new - d).max())
        d = d_new

        hot = (solver_d.element_values(d, m_mean) > REFINE_D).get()
        if (hot & (tree.level < depth)).any():  # the crack reaches coarse elements
            start = time.time()
            old_u, old_d = solver_u.space, solver_d.space
            tree = refine(tree, tree.centers[hot])
            solver_u, solver_d, top, notch, stiffness, h2 = build(tree)
            parent = cp.asarray(old_u.tree.find(tree.level, tree.ij))
            history, driving = history[parent], driving[parent]
            u = solver_u.space.interpolation(old_u) @ u
            d = solver_d.space.interpolation(old_d) @ d
            u_dirichlet = cp.zeros(solver_u.ndof)
            u_dirichlet[top] = load
            w = u - u_dirichlet
            remesh_time += time.time() - start
            print(f"step {step:5d} refined to {len(tree.level)} elements", flush=True)
            continue
        if change < STAGGER_TOL and not switched:
            break
    history = driving
    reactions.append(float(solver_u.apply_full(u)[top].sum()))
    staggers.append(stagger + 1)
    if step % 50 == 0 or stagger > 5:
        print(
            f"step {step:5d} u {load:.2e} reaction {reactions[-1]:.4f} "
            f"staggers {stagger + 1:3d} max d {float(d.max()):.3f} "
            f"elapsed {time.time() - tic:.1f} s",
            flush=True,
        )
    if max(reactions) > 0.1 and reactions[-1] < 0.02 * max(reactions):
        break
toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s ({remesh_time:.2f} s remeshing) for {step + 1} "
    f"steps, {sum(staggers)} staggered iterations, {len(tree.level)} elements, cg "
    f"iterations u mean {np.mean(cg_u):.1f} max {max(cg_u)} d mean {np.mean(cg_d):.1f} "
    f"max {max(cg_d)}, peak reaction {max(reactions):.4f}"
)

# ----------------------------------- postprocessing ----------------------------------
pixels = tree.pixels()
d_elements = solver_d.element_values(d, m_mean).get()
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
ax.imshow(
    np.ma.masked_where(notch[pixels], d_elements[pixels]).T,
    origin="lower",
    cmap="turbo",
    vmin=0.0,
    vmax=1.0,
    extent=[0, LENGTHS[0], 0, LENGTHS[1]],
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
lower = tree.ij * tree.h[:, None]
upper = lower + tree.h[:, None]
edges = np.stack(
    [
        np.stack([lower, np.stack([upper[:, 0], lower[:, 1]], 1)], 1),
        np.stack([lower, np.stack([lower[:, 0], upper[:, 1]], 1)], 1),
    ],
    1,
).reshape(-1, 2, 2)
ax.add_collection(LineCollection(edges, colors="k", linewidths=0.1))
ax.set_xlim(0, LENGTHS[0])
ax.set_ylim(0, LENGTHS[1])
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
ax.plot(loads[: len(reactions)], reactions, "k")
ax.set_xlabel("displacement [mm]")
ax.set_ylabel("reaction force [kN]")
plt.show()
