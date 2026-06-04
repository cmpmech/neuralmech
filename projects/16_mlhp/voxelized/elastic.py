import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../../data"
RESULTS_DIR = BASE_DIR / "../../../results/3D"

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--ct", type=str, default=None)  # only filename
args = parser.parse_args()

# -------------------------------- simulation settings --------------------------------
D = args.dim

DEGREE = 1
ALPHA = 1e-5

E = 210.0
NU = 0.3
FORCE = 1.0

# ------------------------------------ ct geometry ------------------------------------
ct_file = args.ct if args.ct else f"plate_hole_{D}D.npz"
ct = np.load(DATA_DIR / ct_file)
indicator = np.maximum(ct["indicator"].astype(np.float32) / 255.0, ALPHA)

if D == 2:
    Lx, Ly = float(ct["Lx"]), float(ct["Ly"])
    Nx, Ny = indicator.shape
    nelems = [Nx, Ny]
    domain_lengths = [Lx, Ly]
else:
    Lx, Ly, Lz = float(ct["Lx"]), float(ct["Ly"]), float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    nelems = [Nx, Ny, Nz]
    domain_lengths = [Lx, Ly, Lz]

E_vec = mlhp.FloatVector((E * indicator).ravel("C"))  # TODO use uint8
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=nelems, lengths=domain_lengths)
nu_field = mlhp.scalarField(D, NU)

# ---------------------------------------- mesh ---------------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=nelems, lengths=domain_lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=DEGREE, nfields=D)
print(basis)

# -------------------------------- boundary conditions --------------------------------
# uni-axial tension
bc_faces = [0, 2] if D == 2 else [0, 2, 4]  # with x, y, (z) constrained
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2
    )
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)

# -------------------------------------- assembly -------------------------------------
kinematics = mlhp.smallStrainKinematics(D)
constitutive = (
    mlhp.planeStressMaterial(E_field, nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(E_field, nu_field)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, constitutive, mlhp.vectorField(D, [0.0] * D)
)

# one sub-cell per element for preintegrated voxel FEM (E constant per element)
quadrature = mlhp.gridQuadrature(nsubcells=[1] * D)

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
    mlhp.FloatVector(indicator.ravel("C")), nvoxels=nelems, lengths=domain_lengths
)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2] * D)

out = str(RESULTS_DIR / f"elastic_mlhp_{ct_file[:-4]}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
output = mlhp.PVtuOutput(filename=out)
mlhp.basisOutput(basis, postmesh, output, processors)
print(f"VTU written to {out}.pvtu")

# ----------------------------------- postprocessing ----------------------------------
if D == 2:
    result = mlhp.DataAccumulator()
    mlhp.basisOutput(basis, postmesh, result, processors)
    ux = np.array(result.data()[0])[0::2]
    uy = np.array(result.data()[0])[1::2]

    tri = result.triangulation()
    ind_viz = np.array(result.data()[1])
    tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)

    fig, ax = plt.subplots()
    cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
    fig.colorbar(cb)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()
