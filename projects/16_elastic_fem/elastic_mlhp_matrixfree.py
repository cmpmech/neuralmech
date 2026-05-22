import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse.linalg

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
    elem_lengths = [Lx / Nx, Ly / Ny]
else:
    Lz = float(ct["Lz"])
    Nx, Ny, Nz = indicator.shape
    ncells = [Nx, Ny, Nz]
    lengths = [Lx, Ly, Lz]
    elem_lengths = [Lx / Nx, Ly / Ny, Lz / Nz]

# E_values[e] matches basis.locationMaps() element order — same as indicator.ravel("C")
# for structured mlhp grids (verified: findVoxel in spatial.cpp uses C row-major order)
E_values = (E * indicator).ravel("C")

E_vec = mlhp.DoubleVector(E_values)
E_field = mlhp.scalarFieldFromVoxelData(E_vec, nvoxels=ncells, lengths=lengths)
nu_field = mlhp.scalarField(D, nu)

# ------------------------------- mesh + basis --------------------------------
mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=ncells, lengths=lengths))
basis = mlhp.makeHpTrunkSpace(mesh, degree=args.degree, nfields=D)
ndof = basis.ndof()
print(basis)

# Uni-axial tension: each surface constrains its own normal component only.
# face 0 (x-) → ux=0,  face 2 (y-) → uy=0,  face 4 (z-) → uz=0
# Tangential directions are free → Poisson contraction allowed on all faces.
bc_faces = [0, 2] if D == 2 else [0, 2, 4]
bc_list = [
    mlhp.integrateDirichletDofs(
        mlhp.scalarField(D, 0.0), basis, [face], ifield=face // 2
    )
    for face in bc_faces
]
dirichlet = mlhp.combineDirichletDofs(bc_list)
constrained = np.array(dirichlet[0])

# ----------------------------- K_ref (1-element) ----------------------------
# One element with E=1 gives the reference stiffness matrix. DOF ordering in
# K_ref is consistent with basis.locationMaps() because both use the same mlhp basis.
kinematics = mlhp.smallStrainKinematics(D)
mesh1 = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1] * D, lengths=elem_lengths))
basis1 = mlhp.makeHpTrunkSpace(mesh1, degree=args.degree, nfields=D)
c_ref = (
    mlhp.planeStressMaterial(mlhp.scalarField(D, 1.0), nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(mlhp.scalarField(D, 1.0), nu_field)
)
i_ref = mlhp.staticDomainIntegrand(kinematics, c_ref, mlhp.vectorField(D, [0.0] * D))
de = mlhp.combineDirichletDofs([])
m_ref = mlhp.allocateSparseMatrix(basis1, de[0])
v_ref = mlhp.allocateRhsVector(m_ref)
mlhp.integrateOnDomain(
    basis1,
    i_ref,
    [m_ref, v_ref],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
    dirichletDofs=de,
)
K_ref = np.array(m_ref.todense())  # (ndof_e, ndof_e)
K_ref_diag = np.diag(K_ref)

# ----------------------------- RHS assembly ---------------------------------
# Matrix is only used as a scaffold to obtain the interior-DOF RHS vector.
c_full = (
    mlhp.planeStressMaterial(E_field, nu_field)
    if D == 2
    else mlhp.isotropicElasticMaterial(E_field, nu_field)
)
i_full = mlhp.staticDomainIntegrand(kinematics, c_full, mlhp.vectorField(D, [0.0] * D))
matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)

tic = time.time()
mlhp.integrateOnDomain(
    basis,
    i_full,
    [matrix, vector],
    quadrature=mlhp.gridQuadrature(nsubcells=[1] * D),
    dirichletDofs=dirichlet,
)
traction = force / Ly if D == 2 else force / (Ly * Lz)
neumann = mlhp.normalNeumannIntegrand(mlhp.scalarField(D, traction))
right_quad = mlhp.quadratureOnMeshFaces(mesh, [1])
mlhp.integrateOnSurface(basis, neumann, [vector], right_quad, dirichletDofs=dirichlet)
print(f"RHS assembly: {time.time() - tic:.2f}s")

# Inflate interior RHS to full DOF space
interior_mask = np.ones(ndof, dtype=bool)
interior_mask[constrained] = False
rhs = np.zeros(ndof)
rhs[np.where(interior_mask)[0]] = np.array(list(vector))
del matrix, vector  # discard — not needed for the matrix-free solve

# ----------------------- matrix-free operator + preconditioner --------------
efts = np.array(basis.locationMaps())  # (n_elem, ndof_e)


def matvec(u):
    u_local = u[efts]  # (n_elem, ndof_e)
    Ku_local = E_values[:, None] * (u_local @ K_ref.T)  # (n_elem, ndof_e)
    result = np.zeros(ndof)
    np.add.at(result, efts, Ku_local)
    result[constrained] = u[constrained]  # identity on Dirichlet DOFs
    return result


# Diagonal preconditioner: assemble diag(K) without forming K
diag = np.zeros(ndof)
np.add.at(diag, efts, E_values[:, None] * K_ref_diag[None, :])
diag[constrained] = 1.0

A_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=matvec)
P_op = scipy.sparse.linalg.LinearOperator((ndof, ndof), matvec=lambda v: v / diag)

# --------------------------------- solve ------------------------------------
iters = [0]


def callback(x):
    iters[0] += 1


tic = time.time()
sol, info = scipy.sparse.linalg.cg(
    A_op, rhs, M=P_op, rtol=1e-10, maxiter=20000, callback=callback
)
print(f"CG: {iters[0]} iterations, info={info}, {time.time() - tic:.2f}s")
print(f"max displacement: {np.max(np.abs(sol)):.3e}")

# ----------------------------- postprocessing --------------------------------
all_dofs = mlhp.DoubleVector(sol.tolist())
indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.DoubleVector(indicator.ravel("C")), nvoxels=ncells, lengths=lengths
)
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
        out = BASE_DIR.parent.parent / "results" / "16_elastic_fem_mlhp_mf_ux.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    elif args.animate:
        out = (
            BASE_DIR.parent.parent
            / "results"
            / "animations"
            / "16_elastic_fem_mlhp_mf_ux.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    else:
        plt.show()
else:
    out_stem = str(BASE_DIR / "output" / "elastic_mlhp_mf_3d")
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    output = mlhp.PVtuOutput(filename=out_stem)
    mlhp.basisOutput(basis, postmesh, output, processors)
    print(f"VTU written to {out_stem}.pvtu")
