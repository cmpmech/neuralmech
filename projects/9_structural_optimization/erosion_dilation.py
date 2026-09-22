import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

from solvers.optimization import DensityFilter, StructuredFEM, dsimp, simp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# cantilever clamped on the left, loaded downward at the middle of the right edge
# geometry
LENGTHS = [5.0, 1.0]
LOAD_WIDTH = 0.03  # traction patch at the middle of the right edge

# discretization
NELY = 60
DEGREE = 1
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.4
PENAL = 3.0
RMIN = 5.0  # in elements
E0, EMIN, NU = 1.0, 1e-9, 0.3

# optimization (optimality criterion)
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 300
CHANGE_TOL = 0.01

# thresholding
ETA_E, ETA_I, ETA_D = 0.6, 0.5, 0.4  # eroded / intermediate / dilated thresholds
UPSAMPLE = 4  # linear interpolation before thresholding, smooths the exported boundary

# ---------------------------------------- setup --------------------------------------
nelx = int(LENGTHS[0] / LENGTHS[1]) * NELY
elem_lengths = [LENGTHS[0] / nelx, LENGTHS[1] / NELY]

mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx, NELY], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=2)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=2)
material = mlhp.planeStressMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
K_locals = mlhp.integratePartitionMatrices(
    basis_local,
    integrand,
    mlhp.gridQuadrature(nsubcells=[1, 1]),
    mlhp.absoluteQuadratureOrder([QUAD_ORDER, QUAD_ORDER]),
)


# faces: 0=left, 1=right, 2=bottom, 3=top
def face_values(face, ifield, function):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, function), basis, [face], ifield=ifield
    )
    dofs, values = mlhp.combineDirichletDofs([bc])
    return np.array(dofs), np.array(values)


patch = f"exp(-((y - {LENGTHS[1] / 2}) / {LOAD_WIDTH})**2)"
fixed = np.concatenate([face_values(0, 0, "0")[0], face_values(0, 1, "0")[0]])
load_dofs, load_values = face_values(1, 1, patch)

free = np.setdiff1d(np.arange(ndof), fixed)
force = np.zeros(ndof)
force[load_dofs] = -load_values / load_values.sum()
force_free = force[free]

fem = StructuredFEM(efts, free, ndof, K_locals, (nelx, NELY))
density_filter = DensityFilter(RMIN, (nelx, NELY))

# ------------------------------------ optimization -----------------------------------
x = np.full((nelx, NELY), VOLFRAC)
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    rho = density_filter(x)
    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho, PENAL, EMIN, E0), force_free)
    compliance = force @ u

    dc = density_filter.adjoint(-dsimp(rho, PENAL, EMIN, E0) * fem.element_energy(u))
    dv = density_filter.adjoint(np.ones_like(x))

    base = x * np.maximum(0.0, -dc / dv) ** DAMPING
    lo = np.maximum(0.0, x - MOVE)
    hi = np.minimum(1.0, x + MOVE)
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        x_new = np.clip(base / lmid**DAMPING, lo, hi)
        if density_filter(x_new).mean() > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid

    change = np.abs(x_new - x).max()
    x = x_new
    pbar.set_postfix({"c": f"{compliance:.5g}", "change": f"{change:.2e}"})
    if change < CHANGE_TOL:
        break

# ----------------------------------- postprocessing ----------------------------------
filtered = density_filter(x)
names = ["erosion_dilation_eroded", "erosion_dilation_intermediate",
         "erosion_dilation_dilated"]
fields = [(filtered > eta).astype(float) for eta in [ETA_E, ETA_I, ETA_D]]
for name, field in zip(names, fields):
    u = np.zeros(ndof)
    u[free] = fem.solve(simp(field, PENAL, EMIN, E0), force_free)
    print(f"{name:<31}  compliance {force @ u:.4g}  vol {field.mean():.3f}")

images = [(ndi.zoom(filtered, UPSAMPLE, order=1) > eta).astype(float)
          for eta in [ETA_E, ETA_I, ETA_D]]

aspect = LENGTHS[0] / LENGTHS[1]
if not args.book:
    fig, axes = plt.subplots(3, 1, figsize=(2 * aspect, 6))
    for ax, field in zip(axes, images):
        ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, field in zip(names, images):
        fig, ax = plt.subplots(figsize=(aspect, 1))
        ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf")
        plt.close(fig)
