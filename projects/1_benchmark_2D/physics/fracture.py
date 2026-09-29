import time

import mlhp
import mlhp.mkl
import numpy as np
import torch

from helper import (
    SETTINGS,
    PixelMesh,
    benchmark_parser,
    interpolate,
    load_phase,
    load_setup,
    plot_field,
    save_solution,
    voxel_field,
)
from solvers.material_subroutines.phasefield import ABI, build

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = benchmark_parser(setup="opening")
MATERIAL = SETTINGS["materials"]["fracture"]
parser.add_argument("--E-min", type=float, default=MATERIAL["E"][0])  # kN / mm^2
parser.add_argument("--E-max", type=float, default=MATERIAL["E"][1])
parser.add_argument("--gc-min", type=float, default=MATERIAL["gc"][0])  # kN / mm
parser.add_argument("--gc-max", type=float, default=MATERIAL["gc"][1])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# physics (plane strain, long side of 1 mm)
NU = MATERIAL["nu"]
ELL = MATERIAL["ell"]  # phase-field length scale
RESIDUAL = 1e-7  # stiffness left in fully broken material

# loading: the setup's prescribed values ramped in equal steps until the body breaks
NSTEPS = 100

# solver
STAGGER_TOL = 1e-4  # max damage change between staggered iterations
MAX_STAGGER = 500

# ------------------------------------- load data -------------------------------------
g = load_phase(args)
bc, sources = load_setup(args)
E = interpolate(g, args.E_min, args.E_max)
E_field = voxel_field(E)
Gc_field = voxel_field(interpolate(g, args.gc_min, args.gc_max))

# --------------------------------------- setup ---------------------------------------
lib = build()
mesh_u = PixelMesh(args.resolution, nfields=2, aspect=args.aspect)
mesh_d = PixelMesh(args.resolution, nfields=1, aspect=args.aspect)
kinematics = mlhp.smallStrainKinematics(2)
fixed = mesh_u.dirichlet(bc)
ids = np.array(fixed[0], dtype=int)
loaded = ids[np.abs(fixed[1]) > 0]  # dofs whose reaction is recorded

# solution vectors live in mlhp, the numpy views update the fields built on them
u_dofs = mlhp.DoubleVector(mesh_u.ndof, 0.0)
d_dofs = mlhp.DoubleVector(mesh_d.ndof, 0.0)
u, d = np.asarray(u_dofs.buffer), np.asarray(d_dofs.buffer)

damage = mlhp.meshFunction(mesh_d.basis, d_dofs)
material = mlhp.constitutiveEquation(
    2,
    lib.material_address,
    fields=[damage, E_field],
    symmetric=True,
    data=[[args.E_max, NU, RESIDUAL]],
    abi=ABI,
)
equilibrium = mlhp.staticDomainIntegrand(kinematics, material, dofs=u_dofs)

# allocated once, refilled every iteration
matrix_u = mlhp.allocateSparseMatrix(mesh_u.basis, fixed[0])
vector_u = mlhp.allocateRhsVector(matrix_u)
matrix_d = mlhp.allocateSparseMatrix(mesh_d.basis)
vector_d = mlhp.allocateRhsVector(matrix_d)
matrix_r = mlhp.allocateSparseMatrix(mesh_u.basis)
reaction = mlhp.allocateRhsVector(matrix_r)

# ------------------------------------- simulation ------------------------------------
history = mlhp.meshFunction(mesh_u.grid, [0.0])
reactions = []
staggers = 0

tic = time.time()
for step in range(NSTEPS):
    factor = (step + 1) / NSTEPS
    target = np.zeros(mesh_u.ndof)
    target[ids] = factor * np.array(fixed[1])
    load = mesh_u.neumann(bc, factor) + mesh_u.sources(sources, g, factor)

    # driving force max(H, psi+), evaluated lazily from the current displacement
    driving = mlhp.meshFunctionStrainUpdate(
        history,
        mesh_u.basis,
        mesh_u.basis,
        u_dofs,
        u_dofs,
        lib.history_address,
        kinematics=kinematics,
        fields=[E_field],
        data=[[args.E_max, NU]],
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
        dirichlet = [fixed[0], (target - u)[ids].tolist()]
        mlhp.fill(matrix_u)
        mlhp.fill(vector_u)
        mlhp.integrateOnDomain(
            mesh_u.basis, equilibrium, [matrix_u, vector_u], dirichletDofs=dirichlet
        )
        residual = mlhp.DoubleVector(
            (np.asarray(vector_u.buffer) + mesh_u.internal(load, fixed)).tolist()
        )
        with mlhp.mkl.pardisoFactorize(matrix_u, symmetric=True) as solve:
            u += np.asarray(mlhp.inflateDofs(solve(residual), dirichlet).buffer)

        mlhp.fill(matrix_d)
        mlhp.fill(vector_d)
        mlhp.integrateOnDomain(mesh_d.basis, crack, [matrix_d, vector_d])
        with mlhp.mkl.pardisoFactorize(matrix_d, symmetric=True) as solve:
            d_new = np.array(solve(vector_d).buffer)
        change = np.abs(d_new - d).max()
        d[:] = d_new
        if change < STAGGER_TOL:
            break
    history = mlhp.localL2Projection(mesh_u.grid, driving, 1)
    staggers += stagger + 1

    # reaction on the loaded dofs from the unconstrained residual
    mlhp.fill(matrix_r)
    mlhp.fill(reaction)
    mlhp.integrateOnDomain(mesh_u.basis, equilibrium, [matrix_r, reaction])
    reactions.append(-np.asarray(reaction.buffer)[loaded].sum())
    if step % 10 == 0 or stagger > 20:
        print(
            f"step {step:3d} reaction {reactions[-1]:.4f} staggers {stagger + 1:3d} "
            f"max d {d.max():.3f} elapsed {time.time() - tic:.0f} s",
            flush=True,
        )
    if reactions[-1] < 0.1 * max(reactions):  # broken, residual ligaments aside
        break
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s, {staggers} staggers, peak {max(reactions):.4f}")

# ----------------------------------- postprocessing ----------------------------------
field = np.clip(mesh_d.nodal(d)[..., 0], 0.0, 1.0)
if args.save:
    save_solution(
        args,
        "fracture",
        displacement=mesh_u.nodal(u),
        damage=field,
        reactions=np.array(reactions),
        E=E,
    )
plot_field(field, g, "magma_r", "benchmark_fracture", args.book, vmin=0.0, vmax=1.0)
