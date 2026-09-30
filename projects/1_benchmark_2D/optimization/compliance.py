import mlhp
import numpy as np
import torch

from helper import (
    SETTINGS,
    PixelMesh,
    load_setup,
    optimization_parser,
    optimize,
    plot_frames,
    save_designs,
)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = optimization_parser(setup="cantilever")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MATERIAL = SETTINGS["optimization"]["compliance"]
E_MIN, E_MAX = MATERIAL["E"]
NU = MATERIAL["nu"]  # plane strain
VOLUME = MATERIAL["volume"]

# ------------------------------------- load data -------------------------------------
bc, sources = load_setup(args)

# --------------------------------------- setup ---------------------------------------
mesh = PixelMesh(args.resolution, nfields=2, aspect=args.aspect)
material = mlhp.planeStrainMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
force = mesh.neumann(bc) + mesh.sources(sources, np.ones(mesh.shape))

# ------------------------------------ optimization -----------------------------------
designs, compliance = optimize(mesh, integrand, bc, force, VOLUME, E_MIN, E_MAX)

# ----------------------------------- postprocessing ----------------------------------
if args.save:
    save_designs(args, "compliance", designs, compliance)
plot_frames(designs, "benchmark_compliance", args.book)
