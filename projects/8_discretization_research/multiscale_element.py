import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

from helper import Analysis, MultiscaleAnalysis, classic_map, load_patch, optimize

# -------------------------------------- settings -------------------------------------
# multiscale elements; the fake stiffness moves onto the element edges
# discretization
RESOLUTION = 120  # voxels across the height
DISCRETIZATIONS = [(8, 1), (8, 2)]  # (S, p) of the edge traces

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
for sub, degree in DISCRETIZATIONS:
    analysis = MultiscaleAnalysis(nvoxels, sub, degree)
    design_map = classic_map(nvoxels, solid, RMIN)
    rho, _, (t_map, t_fem) = optimize(analysis, design_map, VOLFRAC, ITERS, BETAS)
    rho_thresh = (rho > 0.5).astype(float)
    c_model = analysis.compliance(rho_thresh)[0]
    c_true = reference.compliance(rho_thresh)[0]
    print(
        f"S {sub:2d} p {degree}  ndof {analysis.ndof:6d}  c_model {c_model:.3e}  "
        f"c_true {c_true:.3e}  ratio {c_true / c_model:.2e}  fem {t_fem:.2f} s"
    )
    designs.append(rho)

# ----------------------------------- postprocessing ----------------------------------
fig, axs = plt.subplots(len(designs), 1, figsize=(4, 2 * len(designs)))
for ax, rho, (sub, _) in zip(axs, designs, DISCRETIZATIONS):
    ax.imshow(rho.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.axis("off")
    for edge in range(0, rho.shape[0] + 1, sub):
        ax.axvline(edge - 0.5, color="r", lw=0.2)
    for edge in range(0, rho.shape[1] + 1, sub):
        ax.axhline(edge - 0.5, color="r", lw=0.2)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.05)
plt.show()
