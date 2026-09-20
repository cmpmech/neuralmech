import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../../results/3D").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
D = args.dim

DEGREE = 10
ALPHA = 1e-4
SUB_VOXELS = 40

E = 210.0
NU = 0.3
FORCE = 1.0

# ------------------------------------ ct geometry ------------------------------------
ct_file = args.ct if args.ct else f"plate_hole_{D}D.npz"
ct = np.load(DATA_DIR / ct_file)
indicator = ct["indicator"]  # uint8

if D == 2:
    Lx, Ly = float(ct["Lx"]), float(ct["Ly"])
    Nx, Ny = indicator.shape
    nvoxels = [Nx, Ny]
    domain_lengths = [Lx, Ly]
    voxel_lengths = [Lx / Nx, Ly / Ny]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    nvoxels = [Nx, Ny, Nz]
    domain_lengths = [Lx, Ly, Lz]
    voxel_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

assert all(n % SUB_VOXELS == 0 for n in nvoxels), (
    f"grid dims {nvoxels} must all be divisible by N_VOXELS_PER_ELEM={SUB_VOXELS}"
)

nelems = [n // SUB_VOXELS for n in nvoxels]

# E-field passed to the moment-fitting quadrature (material uses E=1)
E_float = np.maximum(indicator.astype(np.float32) / 255.0, ALPHA)
E_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector((E * E_float).ravel("C")),
    nvoxels=nvoxels,
    lengths=domain_lengths,
)
nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems, lengths=domain_lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
print(basis)

# -------------------------------- boundary conditions --------------------------------
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2
    )
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)

# -------------------------------------- assembly -------------------------------------
kinematics = mlhp.smallStrainKinematics(D)
# E=1 in material: the E-weighting is encoded in the quadrature weights
material = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, material, mlhp.vectorField(D, [0.0] * D)
)

quadrature = mlhp.voxelMomentFittingQuadrature([SUB_VOXELS] * D, E_field, DEGREE)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

tic = time.time()
mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

traction = FORCE / Ly if D == 2 else FORCE / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
print(f"assembly: {time.time() - tic:.2f}s")

# --------------------------------------- solve ---------------------------------------
P = mlhp.diagonalPreconditioner(matrix)
tic = time.time()
interior_dofs, residuals = mlhp.cg(
    matrix, vector, M=P, rtol=1e-10, maxiter=20000, residualNorms=True
)
print(f"CG: {len(residuals)} iterations, {time.time() - tic:.2f}s")
all_dofs = mlhp.inflateDofs(interior_dofs, dirichlet)
print(f"max displacement: {max(abs(v) for v in all_dofs):.3e}")

# --------------------------------------- export --------------------------------------
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(indicator.ravel("C").astype(np.float32) / 255.0),
    nvoxels=nvoxels,
    lengths=domain_lengths,
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)

out = str(RESULTS_DIR / f"elastic_mlhp_momentfitting_{ct_file[:-4]}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out}.pvtu")

# ----------------------------------- postprocessing ----------------------------------
if D == 2:
    result = mlhp.DataAccumulator()
    mlhp.basisOutput(basis, postmesh, result, processors)
    ux = np.array(result.data()[0])[0::2]

    tri = result.triangulation()
    ind_viz = np.array(result.data()[1])
    tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)

    fig, ax = plt.subplots()
    cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
    fig.colorbar(cb)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
