import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
from tqdm import tqdm

from solvers.optimization import DensityFilter, StructuredFEM, dsimp, simp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# half MBB beam (left edge is the symmetry plane) as in mesh_dependence.py, again
# filtered at the element scale, but now at a fixed mesh and with the penalization
# exponent varied
# geometry
LENGTHS = [3.0, 1.0]

# discretization
RESOLUTION = 160  # elements across the height
DEGREE = 1
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.5
PENALIZATIONS = [1.0, 3.0]  # 1 is the linear law, i.e. no penalization
RMIN = 1.5  # filter radius in elements, not in length; 1.2 is the smallest that filters
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# optimization (optimality criterion)
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 500
CHANGE_TOL = 0.001

# postprocessing
THRESHOLD = 0.5


# --------------------------------------- helper --------------------------------------
def optimize(penal):
    nely = RESOLUTION
    nelx = int(round(LENGTHS[0] / LENGTHS[1])) * nely
    elem_lengths = [LENGTHS[0] / nelx, LENGTHS[1] / nely]

    mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx, nely], lengths=LENGTHS))
    basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=2)
    ndof = basis.ndof()
    efts = np.array(basis.locationMaps())

    mesh_local = mlhp.makeRefinedGrid(
        mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths)
    )
    basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=2)
    material = mlhp.planeStressMaterial(
        mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU)
    )
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
    def face_dofs(face, ifield):
        bc = mlhp.integrateDirichletDofs(
            mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
        )
        return np.array(mlhp.combineDirichletDofs([bc])[0])

    symmetry = face_dofs(0, 0)
    roller = np.intersect1d(face_dofs(2, 1), face_dofs(1, 1))
    load_dof = np.intersect1d(face_dofs(3, 1), face_dofs(0, 1))

    fixed = np.unique(np.concatenate([symmetry, roller]))
    free = np.setdiff1d(np.arange(ndof), fixed)

    force = np.zeros(ndof)
    force[load_dof] = LOAD
    force_free = force[free]

    fem = StructuredFEM(efts, free, ndof, K_locals, (nelx, nely))
    density_filter = DensityFilter(RMIN, (nelx, nely))

    rho = np.full((nelx, nely), VOLFRAC)
    pbar = tqdm(range(MAX_ITER), desc=f"p = {penal:.0f}")
    for it in pbar:
        u = np.zeros(ndof)
        u[free] = fem.solve(simp(rho, penal, EMIN, E0), force_free)
        compliance = force @ u

        dc = -dsimp(rho, penal, EMIN, E0) * fem.element_energy(u)
        dc = density_filter.sensitivity(rho, dc)

        base = rho * np.maximum(0.0, -dc) ** DAMPING
        lo = np.maximum(0.0, rho - MOVE)
        hi = np.minimum(1.0, rho + MOVE)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            rho_new = np.clip(base / lmid**DAMPING, lo, hi)
            if rho_new.mean() > VOLFRAC:
                l1 = lmid
            else:
                l2 = lmid

        change = np.abs(rho_new - rho).max()
        rho = rho_new
        pbar.set_postfix({"c": f"{compliance:.5g}", "change": f"{change:.2e}"})

        if change < CHANGE_TOL:
            break

    rho_thresh = (rho > THRESHOLD).astype(float)
    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho_thresh, penal, EMIN, E0), force_free)
    compliance_thresh = force @ u

    # fraction of intermediate density (Sigmund's measure of non-discreteness): zero for
    # a black-and-white design, one for a design that is gray everywhere
    gray = np.mean(4.0 * rho * (1.0 - rho))

    print(
        f"p {penal:.0f}  c {compliance:.5g}  c_thresh {compliance_thresh:.5g}  "
        f"gray {gray:.3f}  vol {rho.mean():.3f} -> {rho_thresh.mean():.3f}  iter {it}"
    )
    return rho


# ------------------------------------ optimization -----------------------------------
designs = [optimize(penal) for penal in PENALIZATIONS]

# ----------------------------------- postprocessing ----------------------------------
aspect = LENGTHS[0] / LENGTHS[1]

if not args.book:
    fig, axes = plt.subplots(len(designs), 1, figsize=(3 * aspect, 3 * len(designs)))
    for ax, penal, design in zip(np.atleast_1d(axes), PENALIZATIONS, designs):
        ax.imshow(design.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
        ax.set_title(f"p = {penal:.0f}")
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for penal, design in zip(PENALIZATIONS, designs):
        fig, ax = plt.subplots(figsize=(3 * aspect, 3))
        ax.imshow(design.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"simp{penal:.0f}.pdf")
        plt.close(fig)
