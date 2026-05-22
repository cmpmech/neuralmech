import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

BASE_DIR = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--ct", type=str, default=None)
args = parser.parse_args()

D = args.dim

E = 210.0
nu = 0.3
force = 1.0

# ----------------------------- CT geometry ----------------------------------
ct_default = BASE_DIR.parent.parent / "data" / f"CT_{D}D.npz"
ct = np.load(args.ct or ct_default)
indicator = ct["indicator"]
Lx = float(ct["Lx"])
Ly = float(ct["Ly"])

if D == 2:
    Nx, Ny = indicator.shape
    ncells = [Nx, Ny]
    lengths = [Lx, Ly]
else:
    Lz = float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]

E_vec = mlhp.DoubleVector((E * indicator).ravel("C"))
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=ncells, lengths=lengths)
nu_field = mlhp.scalarField(D, nu)

# ------------------------------- mesh + basis --------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells, lengths=lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=args.degree, nfields=D)
print(basis)

# ---------------------------------- BCs -------------------------------------
# Uni-axial tension: each surface constrains its own normal component only.
# face 0 (x-) → ux=0,  face 2 (y-) → uy=0,  face 4 (z-) → uz=0
# Tangential directions are free → Poisson contraction allowed on all faces.
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2)
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)

# ----------------------------- assembly -------------------------------------
kinematics = mlhp.smallStrainKinematics(D)
constitutive = (
    mlhp.planeStressMaterial(E_field, nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(E_field, nu_field)
)
integrand = mlhp.staticDomainIntegrand(
    kinematics, constitutive, mlhp.vectorField(D, [0.0] * D)
)

# one sub-cell per element = preintegrated voxel FEM (E constant per element)
quadrature = mlhp.gridQuadrature(nsubcells=[1] * D)

matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

tic = time.time()
mlhp.integrateOnDomain(
    basis, integrand, [matrix, vector], quadrature=quadrature, dirichletDofs=dirichlet
)

traction = force / Ly if D == 2 else force / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
print(f"assembly: {time.time() - tic:.2f}s")

# --------------------------------- solve ------------------------------------
P = mlhp.diagonalPreconditioner(matrix)
tic = time.time()
interior_dofs, residuals = mlhp.cg(
    matrix, vector, M=P, rtol=1e-10, maxiter=20000, residualNorms=True
)
print(f"CG: {len(residuals)} iterations, {time.time() - tic:.2f}s")
all_dofs = mlhp.inflateDofs(interior_dofs, dirichlet)
print(f"max displacement: {max(abs(v) for v in all_dofs):.3e}")

# ----------------------------- postprocessing --------------------------------
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.DoubleVector(indicator.ravel("C")), nvoxels=ncells, lengths=lengths)
processors = [
    mlhp.solutionProcessor(D, all_dofs, "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([args.degree + 2] * D)

if D == 2:
    result = mlhp.DataAccumulator()
    mlhp.basisOutput(basis, postmesh, result, processors)
    disp = np.array(result.data()[0])
    ux = disp[0::2]  # interleaved [ux0, uy0, ux1, uy1, ...]

    tri = result.triangulation()
    ind_viz = np.array(result.data()[1])
    tri.set_mask(ind_viz[tri.triangles].mean(axis=1) < 0.5)

    fig, ax = plt.subplots()
    cb = ax.tricontourf(tri, ux, cmap="turbo", levels=24)
    fig.colorbar(cb)
    ax.set_aspect("equal")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.minorticks_off()
    fig.tight_layout(pad=0)

    if args.book:
        out = BASE_DIR.parent.parent / "results" / "16_elastic_fem_mlhp_ux.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    elif args.animate:
        out = BASE_DIR.parent.parent / "results" / "animations" / "16_elastic_fem_mlhp_ux.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    else:
        plt.show()
else:
    out_stem = str(BASE_DIR / "output" / "elastic_mlhp_3d")
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    output = mlhp.PVtuOutput(filename=out_stem)
    mlhp.basisOutput(basis, postmesh, output, processors)
    print(f"VTU written to {out_stem}.pvtu")
