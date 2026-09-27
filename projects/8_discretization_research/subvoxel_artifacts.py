import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from tqdm import tqdm

from helper import Analysis, classic_map, load_patch, optimize

# -------------------------------------- settings -------------------------------------
# subvoxel designs under the classic filter, re-analyzed on the voxel grid
# discretization
RESOLUTION = 120  # voxels across the height
DISCRETIZATIONS = [(1, 1), (4, 1), (8, 1), (4, 2), (8, 2), (8, 3), (8, 4)]  # (S, p)

# physics
VOLFRAC = 0.4
RMIN = 1.5  # voxels

# optimization (MMA with projection continuation)
ITERS = 250
BETAS = {0: 1.0, 50: 2.0, 100: 4.0, 150: 8.0, 200: 16.0}

# ------------------------------------ optimization -----------------------------------
nvoxels = (2 * RESOLUTION, RESOLUTION)
reference = Analysis(nvoxels, 1, 2)
solid = load_patch(nvoxels)

designs = []
for sub, degree in tqdm(DISCRETIZATIONS):
    analysis = Analysis(nvoxels, sub, degree)
    rho, _, _ = optimize(
        analysis, classic_map(nvoxels, solid, RMIN), VOLFRAC, ITERS, BETAS
    )
    rho_thresh = (rho > 0.5).astype(float)
    c_model = analysis.compliance(rho_thresh)[0]
    c_true = reference.compliance(rho_thresh)[0]
    tqdm.write(
        f"S {sub:2d} p {degree}  ndof {analysis.ndof:6d}  c_model {c_model:.3e}  "
        f"c_true {c_true:.3e}  ratio {c_true / c_model:.2e}"
    )
    designs.append(rho)

# ----------------------------------- postprocessing ----------------------------------
fig, axs = plt.subplots(len(designs), 1, figsize=(4, 2 * len(designs)))
for ax, rho in zip(axs, designs):
    ax.imshow(rho.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.05)
plt.show()
