import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from helper import MBBAnalysis, classic_map, closing_map, optimize

# -------------------------------------- settings -------------------------------------
# closing_filter.py on the half MBB beam
# discretization
RESOLUTION = 160  # voxels across the height
DISCRETIZATIONS = [(1, 1), (8, 2), (16, 4), (16, 2), (8, 1)]  # (S, p)
REFERENCE_RADII = [4, 8]  # closing radii of the extra fine (S = 1) references

# physics
VOLFRAC = 0.5
RMIN = 1.5  # voxels

# optimization (MMA with projection continuation)
ITERS = 250
BETAS = {0: 1.0, 50: 2.0, 100: 4.0, 150: 8.0, 200: 16.0}


# --------------------------------------- helper --------------------------------------
def run(sub, degree, radius):
    analysis = MBBAnalysis(nvoxels, sub, degree)
    if radius == 0:
        design_map = classic_map(nvoxels, solid, RMIN)
    else:
        design_map = closing_map(nvoxels, solid, radius, RMIN)
    rho, _, (t_map, t_fem) = optimize(analysis, design_map, VOLFRAC, ITERS, BETAS)
    rho_thresh = (rho > 0.5).astype(float)
    c_model = analysis.compliance(rho_thresh)[0]
    c_true = reference.compliance(rho_thresh)[0]
    print(
        f"S {sub:2d} p {degree} R {radius:4.1f}  ndof {analysis.ndof:6d}  "
        f"c_true {c_true:.4f}  ratio {c_true / c_model:.3f}  "
        f"filter {t_map:.2f} s  fem {t_fem:.2f} s"
    )
    return rho


# ------------------------------------ optimization -----------------------------------
nvoxels = (3 * RESOLUTION, RESOLUTION)
reference = MBBAnalysis(nvoxels, 1, 2)
solid = MBBAnalysis.passive(nvoxels)

designs = [run(1, 1, radius) for radius in REFERENCE_RADII]
for sub, degree in DISCRETIZATIONS:
    designs.append(run(sub, degree, 0 if sub == 1 else sub / degree))

# ----------------------------------- postprocessing ----------------------------------
fig, axs = plt.subplots(len(designs), 1, figsize=(4, 2 * len(designs)))
for ax, rho in zip(axs, designs):
    ax.imshow(rho.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.05)
plt.show()
