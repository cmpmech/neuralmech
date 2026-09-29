import cmasher as cmr
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

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = benchmark_parser(setup="conduction")
MATERIAL = SETTINGS["materials"]["poisson"]
parser.add_argument("--kappa-min", type=float, default=MATERIAL["kappa"][0])
parser.add_argument("--kappa-max", type=float, default=MATERIAL["kappa"][1])
args = parser.parse_args()

# ------------------------------------- load data -------------------------------------
g = load_phase(args)
bc, sources = load_setup(args)
kappa = interpolate(g, args.kappa_min, args.kappa_max)

# --------------------------------------- setup ---------------------------------------
mesh = PixelMesh(args.resolution, nfields=1, aspect=args.aspect)
dirichlet = mesh.dirichlet(bc)

matrix = mlhp.allocateSparseMatrix(mesh.basis, dirichlet[0])
vector = mlhp.allocateRhsVector(matrix)
integrand = mlhp.poissonIntegrand(voxel_field(kappa), mlhp.scalarField(2, 0.0))
mlhp.integrateOnDomain(mesh.basis, integrand, [matrix, vector], dirichletDofs=dirichlet)

# --------------------------------------- solve ---------------------------------------
rhs = np.asarray(vector) + mesh.internal(mesh.neumann(bc), dirichlet)
dofs = mlhp.inflateDofs(mlhp.DoubleVector(solve(matrix, rhs).tolist()), dirichlet)
temperature = mesh.nodal(dofs)[..., 0]

# ----------------------------------- postprocessing ----------------------------------
if args.save:
    save_solution(args, "poisson", temperature=temperature, kappa=kappa)
plot_field(temperature, g, cmr.torch, "benchmark_poisson", args.book)
