from pathlib import Path

import mlhp
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
    solve,
    voxel_field,
)
from postprocessing import load_cmap
from solvers.material_subroutines.j2 import ABI, NHISTORY, build

BASE_DIR = Path(__file__).parent
CMAP_DIR = (BASE_DIR / "../../../.cmap").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = benchmark_parser(setup="stretch")
MATERIAL = SETTINGS["materials"]["plasticity"]
parser.add_argument("--E-min", type=float, default=MATERIAL["E"][0])  # MPa
parser.add_argument("--E-max", type=float, default=MATERIAL["E"][1])
parser.add_argument("--yield-min", type=float, default=MATERIAL["yield"][0])
parser.add_argument("--yield-max", type=float, default=MATERIAL["yield"][1])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# physics (plane strain)
NU = MATERIAL["nu"]
HARDENING = MATERIAL["hardening"]  # linear isotropic hardening modulus

# loading: the setup's prescribed values ramped in equal steps
NSTEPS = 10
NEWTON_ITER = 20
NEWTON_TOL = 1e-8

# [E, nu, sigmaY, H, beta]; E and sigmaY are overridden per pixel by the fields
params = [args.E_max, NU, args.yield_max, HARDENING, 0.0]

# ------------------------------------- load data -------------------------------------
g = load_phase(args)
bc, sources = load_setup(args)
E = voxel_field(interpolate(g, args.E_min, args.E_max))
sigma_y = voxel_field(interpolate(g, args.yield_min, args.yield_max))

# --------------------------------------- setup ---------------------------------------
lib = build()
mesh = PixelMesh(args.resolution, nfields=2, aspect=args.aspect)
kinematics = mlhp.smallStrainKinematics(2)
fixed = mesh.dirichlet(bc)
homogeneous = [fixed[0], [0.0] * len(fixed[0])]

# ----------------------------------- load stepping -----------------------------------
history = mlhp.meshFunction(mesh.grid, [0.0] * NHISTORY)
dofs0 = mlhp.DoubleVector(mesh.ndof, 0.0)

for istep in range(NSTEPS):
    factor = (istep + 1) / NSTEPS
    material = mlhp.constitutiveEquation(
        2,
        lib.material_address,
        fields=[history, E, sigma_y],
        symmetric=True,
        incremental=True,
        data=[params],
        abi=ABI,
    )
    load = mesh.neumann(bc, factor) + mesh.sources(sources, g, factor)

    # the first iteration lifts the prescribed increment (consistent predictor)
    prescribed = [fixed[0], (np.array(fixed[1]) / NSTEPS).tolist()]
    dofs1 = dofs0
    for inewton in range(NEWTON_ITER):
        dirichlet = prescribed if inewton == 0 else homogeneous
        matrix = mlhp.allocateSparseMatrix(mesh.basis, fixed[0])
        vector = mlhp.allocateRhsVector(matrix)
        increment = mlhp.add(dofs1, dofs0, -1.0)
        integrand = mlhp.staticDomainIntegrand(kinematics, material, dofs=increment)
        mlhp.integrateOnDomain(
            mesh.basis, integrand, [matrix, vector], dirichletDofs=dirichlet
        )
        residual = np.asarray(vector) + mesh.internal(load, fixed)

        norm1 = np.linalg.norm(residual)

        if inewton == 0:
            norm0 = norm1
        if inewton > 0 and norm1 <= max(NEWTON_TOL * norm0, 1e-11 * args.E_max):
            break
        solution = mlhp.DoubleVector(solve(matrix, residual).tolist())
        dofs1 = mlhp.add(dofs1, mlhp.inflateDofs(solution, dirichlet))

    # commit the converged plastic state into the history
    updated = mlhp.meshFunctionStrainUpdate(
        history,
        mesh.basis,
        mesh.basis,
        dofs0,
        dofs1,
        lib.update_address,
        kinematics=kinematics,
        fields=[E, sigma_y],
        data=[params],
        abi=ABI,
    )
    history = mlhp.localL2Projection(mesh.grid, updated, 1)
    dofs0 = dofs1
    print(f"step {istep + 1:2d} / {NSTEPS}: newton {inewton}, residual {norm1:.2e}")

# ----------------------------------- postprocessing ----------------------------------
displacement = mesh.nodal(dofs1)
plastic_strain = np.maximum(mesh.cell_values(history)[..., 12], 0.0)

if args.save:
    save_solution(
        args,
        "plasticity",
        displacement=displacement,
        plastic_strain=plastic_strain,
        g=g,
    )
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")
plot_field(plastic_strain, g, rainbow, "benchmark_plasticity", args.book)
