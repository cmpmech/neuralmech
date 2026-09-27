import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import mlhp.mkl
import numpy as np

from postprocessing import load_cmap
from solvers.material_subroutines.phasefield import ABI, build

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CMAP_DIR = (BASE_DIR / "../../.cmap").resolve()
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")  # TODO could be animated
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# single edge notched shear (Miehe et al. 2010), units kN and mm
# geometry
LENGTH = 1.0
NOTCH = 0.5  # notch length from the left edge at mid height, two elements thick

# discretization
ELEMENTS = 16  # bilinear elements per edge away from the crack
LEVELS = 3  # refinement levels along the notch and in the lower right quadrant
BAND = 0.1  # half width of the refined region around the notch

# physics (plane strain)
E = 210.0
NU = 0.3
GC = 2.7e-3  # fracture toughness
ELL = 0.015  # phase-field length scale, about two finest elements
RESIDUAL = 1e-7  # stiffness left in fully broken material
ALPHA = 1e-6  # stiffness of the notch elements

# loading: horizontal top edge displacement
LOADS = np.arange(1, 101) * 2e-4

# solver
STAGGER_TOL = 1e-4  # max damage change between staggered iterations
MAX_STAGGER = 500


# --------------------------------------- helper --------------------------------------
def voxel_field(values):
    data = mlhp.DoubleVector(values.ravel().tolist())
    return mlhp.scalarFieldFromVoxelData(data, list(values.shape), [LENGTH] * 2)


# ---------------------------------------- setup --------------------------------------
lib = build()
fine = ELEMENTS * 2**LEVELS
centres = (np.arange(fine) + 0.5) / fine * LENGTH
x, y = np.meshgrid(centres, centres, indexing="ij")
notch = (x < NOTCH) & (np.abs(y - 0.5 * LENGTH) < LENGTH / fine)
refined = (np.abs(y - 0.5 * LENGTH) < BAND) | (x > NOTCH - BAND) & (y < 0.5 + BAND)

grid = mlhp.makeRefinedGrid([ELEMENTS] * 2, [LENGTH] * 2)
grid.refine(mlhp.refineWithLevelFunction(voxel_field(LEVELS * refined), nseedpoints=4))
basis_u = mlhp.makeHpTensorSpace(grid, degree=1, nfields=2)
basis_d = mlhp.makeHpTensorSpace(grid, degree=1, nfields=1)
kinematics = mlhp.smallStrainKinematics(2)
E_field = voxel_field(np.where(notch, ALPHA * E, E))
Gc_field = mlhp.scalarField(2, GC)
print(f"elements {grid.ncells()}, dofs u {basis_u.ndof()} d {basis_d.ndof()}")

# bottom clamped, top sheared and held vertically, sides free
zero, one = mlhp.scalarField(2, 0.0), mlhp.scalarField(2, 1.0)
top = mlhp.integrateDirichletDofs(one, basis_u, [3], ifield=0)[0]
held = [mlhp.integrateDirichletDofs(zero, basis_u, [2], ifield=0)[0]]
held += [mlhp.integrateDirichletDofs(zero, basis_u, [2, 3], ifield=1)[0]]
fixed = np.unique(np.concatenate([top] + held))
top = np.array(top)

# solution vectors live in mlhp, the numpy views update the fields built on them
u_dofs = mlhp.DoubleVector(basis_u.ndof(), 0.0)
d_dofs = mlhp.DoubleVector(basis_d.ndof(), 0.0)
u, d = np.asarray(u_dofs.buffer), np.asarray(d_dofs.buffer)

damage = mlhp.meshFunction(basis_d, d_dofs)
material = mlhp.constitutiveEquation(
    2,
    lib.material_address,
    fields=[damage, E_field],
    symmetric=True,
    data=[[E, NU, RESIDUAL]],
    abi=ABI,
)
equilibrium = mlhp.staticDomainIntegrand(kinematics, material, dofs=u_dofs)

# allocated once, refilled every iteration
matrix_u = mlhp.allocateSparseMatrix(basis_u, fixed.tolist())
vector_u = mlhp.allocateRhsVector(matrix_u)
matrix_d = mlhp.allocateSparseMatrix(basis_d)
vector_d = mlhp.allocateRhsVector(matrix_d)
matrix_r = mlhp.allocateSparseMatrix(basis_u)
reaction = mlhp.allocateRhsVector(matrix_r)

# ------------------------------------- simulation ------------------------------------
history = mlhp.meshFunction(grid, [0.0])
reactions = []
staggers = 0

tic = time.time()
for step, load in enumerate(LOADS):
    target = np.zeros(basis_u.ndof())
    target[top] = load

    # driving force max(H, psi+), evaluated lazily from the current displacement
    driving = mlhp.meshFunctionStrainUpdate(
        history,
        basis_u,
        basis_u,
        u_dofs,
        u_dofs,
        lib.history_address,
        kinematics=kinematics,
        fields=[E_field],
        data=[[E, NU]],
        abi=ABI,
    )
    crack = mlhp.domainIntegrand(
        2,
        lib.damage_address,
        types=[mlhp.AssemblyType.UnsymmetricMatrix, mlhp.AssemblyType.Vector],
        maxdiff=1,
        fields=[driving, Gc_field],
        data=[[ELL]],
        abi=ABI,
    )
    for stagger in range(MAX_STAGGER):
        # one Newton step on the tension/compression split, lifting the load first
        dirichlet = [fixed.tolist(), (target - u)[fixed].tolist()]
        mlhp.fill(matrix_u)
        mlhp.fill(vector_u)
        mlhp.integrateOnDomain(
            basis_u, equilibrium, [matrix_u, vector_u], dirichletDofs=dirichlet
        )
        with mlhp.mkl.pardisoFactorize(matrix_u, symmetric=True) as solve:
            u += np.asarray(mlhp.inflateDofs(solve(vector_u), dirichlet).buffer)

        mlhp.fill(matrix_d)
        mlhp.fill(vector_d)
        mlhp.integrateOnDomain(basis_d, crack, [matrix_d, vector_d])
        with mlhp.mkl.pardisoFactorize(matrix_d, symmetric=True) as solve:
            d_new = np.array(solve(vector_d).buffer)
        change = np.abs(d_new - d).max()
        d[:] = d_new
        if change < STAGGER_TOL:
            break
    history = mlhp.localL2Projection(grid, driving, 1)
    staggers += stagger + 1

    # reaction force on the top edge from the unconstrained residual
    mlhp.fill(matrix_r)
    mlhp.fill(reaction)
    mlhp.integrateOnDomain(basis_u, equilibrium, [matrix_r, reaction])
    reactions.append(-np.asarray(reaction.buffer)[top].sum())
    if step % 10 == 0 or stagger > 20:
        print(
            f"step {step:4d} u {load:.2e} reaction {reactions[-1]:.4f} "
            f"staggers {stagger + 1:3d} max d {d.max():.3f} "
            f"elapsed {time.time() - tic:.0f} s",
            flush=True,
        )
    if max(reactions) > 0.1 and reactions[-1] < 0.02 * max(reactions):
        break
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s, {staggers} staggers, peak {max(reactions):.4f} kN")

# ----------------------------------- postprocessing ----------------------------------
result = mlhp.DataAccumulator()
processors = [mlhp.functionProcessor(damage, "Damage")]
mlhp.basisOutput(
    basis_d, cellmesh=mlhp.gridCellMesh([2, 2]), processors=processors, output=result
)
tri = result.triangulation(mpl=True)
cx, cy = tri.x[tri.triangles].mean(1), tri.y[tri.triangles].mean(1)
tri.set_mask((cx < NOTCH) & (np.abs(cy - 0.5 * LENGTH) < LENGTH / fine))
field = np.clip(np.array(result.data()[0]), 0.0, 1.0)

fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.tricontourf(
    tri, field, levels=np.linspace(0.0, 1.0, 65), cmap="magma_r", extend="both"
)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if not args.book:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(LOADS[: len(reactions)], reactions, "k")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("reaction force [kN]")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    plt.savefig(RGB_PDF_DIR / "fracture2D.pdf", transparent=True)
    plt.close()
