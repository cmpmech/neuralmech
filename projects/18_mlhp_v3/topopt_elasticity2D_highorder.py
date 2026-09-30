import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from highorder import HighOrderMultigrid, skeleton_nodes
from tqdm import tqdm

from solvers.optimization import DensityFilter, dsimp, simp

BASE_DIR = Path(__file__).parent

# -------------------------------------- settings -------------------------------------
# half MBB beam (left edge is the symmetry plane)
# geometry
LENGTHS = [4.0, 1.0]

# discretization
NX, NY = 2560, 640  # divisible by SUB_VOXELS, element counts by powers of two
# Q_p elements over s x s voxels, condensed to their edges. With RMIN = 2 voxels,
# (2, 4), (3, 4) and (3, 5) stay free of artificially stiff patterns at 5120 x 1280
# voxels, (2, 5) and (3, 8) grow stripes in low-stress regions
DEGREE = 2
SUB_VOXELS = 4

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

# ---------------------------------------- helper -------------------------------------
oc_update = cp.ElementwiseKernel(
    "T rho, T base, T scale, T move",
    "T rho_new",
    "rho_new = min(min((T)1, rho + move), max(max((T)0, rho - move), base * scale))",
    "oc_update",
)

# ---------------------------------------- setup --------------------------------------
ex, ey = NX // SUB_VOXELS, NY // SUB_VOXELS
gi, gj = skeleton_nodes(ex, ey, DEGREE)  # positions on the (p ex + 1, p ey + 1) grid
fixed = np.stack([gi == 0, (gi == DEGREE * ex) & (gj == 0)])
force = cp.zeros((2, len(gi)))
force[1, cp.asarray((gi == 0) & (gj == DEGREE * ey))] = LOAD
force = force.ravel()

tic = time.time()
solver = HighOrderMultigrid(
    [NX, NY], LENGTHS, NU, fixed, degree=DEGREE, sub_voxels=SUB_VOXELS
)
print(f"multigrid levels {[level.ndof for level in solver.levels]}")
print(f"setup time {time.time() - tic:.2f} s")

density_filter = DensityFilter(RMIN, (NX, NY), xp=cp, dtype=cp.float32)

# ------------------------------------ optimization -----------------------------------
rho = cp.full((NX, NY), VOLFRAC, dtype=cp.float32)
u = None
cg_history = []

tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    solver.update(simp(rho, PENAL, EMIN, E0))
    u, cg_iters = solver.solve(force, x0=u, rtol=CG_TOL)
    compliance = float(force @ u)

    dc = -dsimp(rho, PENAL, EMIN, E0) * solver.element_energy(u)
    dc = density_filter.sensitivity(rho, dc)

    # optimality criterion update with bisection on the log volume multiplier
    base = rho * cp.maximum(0.0, -dc) ** DAMPING
    move = cp.float32(MOVE)
    l1, l2 = 1e-9, 1e9
    while l2 / l1 > 1.0 + 2e-4:
        lmid = (l1 * l2) ** 0.5
        rho_new = oc_update(rho, base, cp.float32(lmid**-DAMPING), move)
        if float(rho_new.mean(dtype=cp.float64)) > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid

    change = float(cp.abs(rho_new - rho).max())
    rho = rho_new
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
    f"cg iterations mean {np.mean(cg_history):.1f} max {max(cg_history)}\n"
    f"gpu memory {cp.get_default_memory_pool().total_bytes() / (NX * NY):.0f} B/voxel"
)

# ----------------------------------- postprocessing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(cp.float32)
solver.update(simp(rho_thresh, PENAL, EMIN, E0))
u_thresh, _ = solver.solve(force, rtol=CG_TOL)
compliance_thresh = float(force @ u_thresh)
print(f"thresholded  c {compliance_thresh:.3e} vol {float(rho_thresh.mean()):.3f}")

for field in (rho, rho_thresh):
    fig, ax = plt.subplots(figsize=(12.8, 3.2), dpi=150)
    ax.imshow(field.get().T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
