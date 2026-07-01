import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import pypardiso
import scipy.sparse as sp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = (BASE_DIR / "../../results/animations/animation_frames").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
DIM = 2

# discretization
DEGREE = 2
NELEMENTS = 200 if DIM == 1 else 40
NELEMENTS = [NELEMENTS] * DIM

# physics
CONDUCTIVITY = 0.1
CAPACITY = 1.5
LENGTH = 1.0

# time integration (theta=1 backward euler, 0.5 crank-nicolson)
if DIM == 1:
    T = 0.1
    N = 250
else:
    T = 0.3
    N = 250
THETA = 1.0

# gaussian hot-spot initial condition
AMPLITUDE = 1.0
WIDTH = 0.05

# snapshot cadence for the 1d space-time image and animation frames
SAVE_EVERY = 1

# --------------------------------------- setup ---------------------------------------
# implicit: dt is set by accuracy, NOT by the explicit courant limit dx**2 / (2 * kappa)
dt = T / N
center = [0.5 * LENGTH] * DIM
lengths = [LENGTH] * DIM

# ---------------------------------------- mesh ---------------------------------------
grid = mlhp.makeRefinedGrid(NELEMENTS, lengths)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=1)
print(basis)

# --------------------------------------- fields --------------------------------------
# capacity, conductivity and source are space-time (dim + 1) fields; constants broadcast
capacity = mlhp.scalarField(DIM + 1, CAPACITY)
conductivity = mlhp.scalarField(DIM + 1, CONDUCTIVITY)
source = mlhp.scalarField(DIM + 1, 0.0)

coords = ["x", "y", "z"][:DIM]
radius2 = " + ".join(f"({c} - {center[i]})**2" for i, c in enumerate(coords))
initial = mlhp.scalarField(DIM, f"{AMPLITUDE} * exp(-({radius2}) / (2 * {WIDTH}**2))")

# -------------------------------- boundary conditions --------------------------------
# dirichlet-cold on every face
dirichlet = mlhp.integrateDirichletDofs(
    mlhp.scalarField(DIM, 0.0), basis, list(range(2 * DIM))
)

# ----------------------------------- time stepping -----------------------------------
dofs = mlhp.projectOnto(basis, initial)

record_every = SAVE_EVERY if (DIM == 1 or args.animate) else None
recorded = [dofs] if record_every is not None else []

for istep in range(N):
    time0, time1 = istep * dt, (istep + 1) * dt

    # the lhs matrix M / dt + theta * K is identical every step (fixed mesh, dt, material);
    # reassembled here for clarity, but its factorization could be cached for speed
    matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)

    integrand = mlhp.transientPoissonIntegrand(
        capacity, conductivity, source, dofs, [time0, time1], THETA
    )
    mlhp.integrateOnDomain(
        basis, basis, integrand, [matrix, vector], dirichletDofs=dirichlet
    )

    operator = sp.csr_matrix(
        (
            np.asarray(matrix.data_array),
            np.asarray(matrix.indices_array),
            np.asarray(matrix.indptr_array),
        ),
        shape=tuple(matrix.shape),
    )
    internalDofs = pypardiso.spsolve(operator, np.asarray(vector))
    dofs = mlhp.inflateDofs(mlhp.DoubleVector(internalDofs), dirichlet)

    if record_every is not None and (istep + 1) % record_every == 0:
        recorded.append(dofs)


# ----------------------------------- postprocessing ----------------------------------
def sample(dofs):
    result = mlhp.DataAccumulator()
    cellmesh = mlhp.gridCellMesh([DEGREE + 1] * DIM)
    processors = [mlhp.solutionProcessor(DIM, dofs, "Temperature")]
    mlhp.basisOutput(basis, cellmesh=cellmesh, processors=processors, output=result)
    return result


def profile(dofs):
    # sample a 1d field, sorted and de-duplicated along x
    result = sample(dofs)
    x = np.asarray(result.mesh().points()).reshape(-1, 3)[:, 0]
    values = np.asarray(result.data()[0])
    order = np.argsort(x)
    _, unique = np.unique(np.round(x[order], 9), return_index=True)
    return x[order][unique], values[order][unique]


# 3d has no inline preview; write the final field to vtu for paraview
if DIM == 3:
    cellmesh = mlhp.gridCellMesh([DEGREE + 1] * DIM)
    processors = [mlhp.solutionProcessor(DIM, dofs, "Temperature")]
    mlhp.basisOutput(
        basis,
        cellmesh=cellmesh,
        processors=processors,
        output=mlhp.VtuOutput(str(RESULTS_DIR / "heat3D")),
    )
    print(f"wrote {RESULTS_DIR / 'heat3D'}.vtu")

# build the inline / book figure (dim 1 and 2)
if DIM in (1, 2) and not args.animate:
    if DIM == 1:
        spacetime = np.stack([profile(snap)[1] for snap in recorded], axis=1)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pcolormesh(spacetime.T, cmap=cmr.torch)
        ax.axis("off")
    else:
        result = sample(dofs)
        data = np.asarray(result.data()[0])
        # floor the contour levels at exactly 0 so the cold field fills solidly; an
        # integer level count lets matplotlib pick a lowest level above 0, leaving the
        # zero region unfilled (white speckle)
        levels = np.linspace(0, data.max(), 64)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
        ax.tricontourf(
            result.triangulation(mpl=True),
            data,
            levels=levels,
            cmap=cmr.torch,
            extend="max",
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if not args.book:
        plt.show()
# -------------------------------- book postprocessing --------------------------------
    else:
        plt.savefig(RGB_PDF_DIR / f"heat{DIM}D.pdf")
        plt.close()
# ----------------------------------- animate export ----------------------------------
if args.animate and DIM in (1, 2):
    frame_dir = ANIMATION_DIR / f"heat{DIM}D"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if DIM == 2:
        # one constant color scale across frames (so the decay reads) with levels
        # floored at 0; per-frame integer levels leave the cold field unfilled (white)
        scale = max(np.asarray(sample(snap).data()[0]).max() for snap in recorded)
        levels = np.linspace(0, scale, 64)
    for i, snap in enumerate(recorded):
        fig, ax = plt.subplots(figsize=(5, 5))
        if DIM == 1:
            x, values = profile(snap)
            ax.plot(x, values, "k")
            ax.set_ylim(0, AMPLITUDE)
        else:
            # each sample orders its points differently, so re-triangulate per frame
            result = sample(snap)
            ax.tricontourf(
                result.triangulation(mpl=True),
                result.data()[0],
                levels=levels,
                cmap=cmr.torch,
                extend="min",
            )
            ax.set_aspect("equal")
            ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(frame_dir / f"frame_{i}.jpg")
        plt.close()
