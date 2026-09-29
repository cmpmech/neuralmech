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
    pixel_gradient,
    plot_field,
    save_solution,
    solve,
    voxel_field,
)
from postprocessing import load_cmap

BASE_DIR = Path(__file__).parent
CMAP_DIR = (BASE_DIR / "../../../.cmap").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = benchmark_parser(setup="tension")
MATERIAL = SETTINGS["materials"]["elasticity"]
parser.add_argument("--E-min", type=float, default=MATERIAL["E"][0])
parser.add_argument("--E-max", type=float, default=MATERIAL["E"][1])
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
NU = MATERIAL["nu"]  # plane strain

# ------------------------------------- load data -------------------------------------
g = load_phase(args)
bc, sources = load_setup(args)
E = interpolate(g, args.E_min, args.E_max)

# --------------------------------------- setup ---------------------------------------
mesh = PixelMesh(args.resolution, nfields=2, aspect=args.aspect)
dirichlet = mesh.dirichlet(bc)

matrix = mlhp.allocateSparseMatrix(mesh.basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
material = mlhp.planeStrainMaterial(voxel_field(E), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
mlhp.integrateOnDomain(mesh.basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

# --------------------------------------- solve ---------------------------------------
load = mesh.neumann(bc) + mesh.sources(sources, g)
rhs = np.asarray(vector) + mesh.internal(load, dirichlet)
dofs = mlhp.inflateDofs(mlhp.DoubleVector(solve(matrix, rhs).tolist()), dirichlet)
displacement = mesh.nodal(dofs)

# ----------------------------------- postprocessing ----------------------------------
# von Mises stress per pixel, plane strain
grad = pixel_gradient(displacement)
exx, eyy = grad[..., 0, 0], grad[..., 1, 1]
exy = 0.5 * (grad[..., 0, 1] + grad[..., 1, 0])
lam, mu = E * NU / ((1 + NU) * (1 - 2 * NU)), 0.5 * E / (1 + NU)
sxx, syy = lam * (exx + eyy) + 2 * mu * exx, lam * (exx + eyy) + 2 * mu * eyy
szz, sxy = NU * (sxx + syy), 2 * mu * exy
von_mises = np.sqrt(
    0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy**2
)

if args.save:
    save_solution(
        args, "elasticity", displacement=displacement, von_mises=von_mises, E=E
    )
rainbow = load_cmap(CMAP_DIR / "rainbow_desaturated.cmap")
plot_field(
    von_mises,
    g,
    rainbow,
    "benchmark_elasticity",
    args.book,
    vmax=np.percentile(von_mises, 99),
)
