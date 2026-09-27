import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import mlhp
import numpy as np
from tqdm import tqdm

from helper import VoxelMultigrid
from solvers.optimization import DensityFilter, dsimp, simp

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# half MBB beam (left edge is the symmetry plane)
# geometry
LENGTHS = [4.0, 1.0]

# discretization
NX, NY = 640, 160
DEGREE = 1
SUB_VOXELS = 1  # design voxels per element edge

# physics
VOLFRAC = 0.5
PENAL = 3.0
RMIN = 2
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# postprocessing
THRESHOLD = 0.5

# optimization (optimality criterion)
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 500
CHANGE_TOL = 0.01

# solver
CG_TOL = 1e-5  # relative residual, sensitivities to ~1e-6


# -------------------------------- boundary conditions --------------------------------
# faces: 0=left, 1=right, 2=bottom, 3=top
def face_dofs(basis, face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


def mbb_supports(basis):
    symmetry = face_dofs(basis, 0, 0)
    roller = np.intersect1d(face_dofs(basis, 2, 1), face_dofs(basis, 1, 1))
    return np.unique(np.concatenate([symmetry, roller]))


# ---------------------------------------- setup --------------------------------------
material = mlhp.planeStressMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
tic = time.time()
solver = VoxelMultigrid(
    integrand,
    [NX, NY],
    LENGTHS,
    DEGREE,
    SUB_VOXELS,
    2,
    mbb_supports,
)
print(f"multigrid levels {[level.ndof for level in solver.levels]}")
print(f"setup time {time.time() - tic:.2f} s")

load_dof = np.intersect1d(face_dofs(solver.basis, 3, 1), face_dofs(solver.basis, 0, 1))
force = cp.zeros(solver.ndof)
force[cp.asarray(load_dof)] = LOAD

density_filter = DensityFilter(RMIN, (NX, NY), xp=cp)

# ------------------------------------ optimization -----------------------------------
rho = cp.full((NX, NY), VOLFRAC)
u = None
history = []
cg_history = []

tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    solver.update(simp(rho, PENAL, EMIN, E0))
    u, cg_iters = solver.solve(force, x0=u, rtol=CG_TOL)
    compliance = float(force @ u)

    dc = -dsimp(rho, PENAL, EMIN, E0) * solver.element_energy(u)
    dc = density_filter.sensitivity(rho, dc)

    # optimality criterion update with bisection on the volume multiplier
    base = rho * cp.maximum(0.0, -dc) ** DAMPING
    lo = cp.maximum(0.0, rho - MOVE)
    hi = cp.minimum(1.0, rho + MOVE)
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = cp.clip(base / lmid**DAMPING, lo, hi)
        if float(rho_new.mean()) > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid

    change = float(cp.abs(rho_new - rho).max())
    rho = rho_new
    history.append(compliance)
    cg_history.append(cg_iters)
    pbar.set_postfix(
        {"c": f"{compliance:.3e}", "change": f"{change:.2e}", "cg": cg_iters}
    )

    if change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s\n"
    f"cg iterations mean {np.mean(cg_history):.1f} max {max(cg_history)}"
)

# ----------------------------------- postprocessing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)
solver.update(simp(rho_thresh, PENAL, EMIN, E0))
u_thresh, _ = solver.solve(force, rtol=CG_TOL)
compliance_thresh = float(force @ u_thresh)
print(f"thresholded  c {compliance_thresh:.3e} vol {float(rho_thresh.mean()):.3f}")

for field in (rho, rho_thresh):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.imshow(field.get().T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
