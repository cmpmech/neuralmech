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

parser = optimization_parser(setup="heat_sink")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MATERIAL = SETTINGS["optimization"]["heat_conduction"]
KAPPA_MIN, KAPPA_MAX = MATERIAL["kappa"]
VOLUME = MATERIAL["volume"]

# ------------------------------------- load data -------------------------------------
bc, sources = load_setup(args)

# --------------------------------------- setup ---------------------------------------
mesh = PixelMesh(args.resolution, nfields=1, aspect=args.aspect)
integrand = mlhp.poissonIntegrand(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, 0.0))
force = mesh.neumann(bc) + mesh.sources(sources, np.ones(mesh.shape))

# ------------------------------------ optimization -----------------------------------
designs, compliance = optimize(mesh, integrand, bc, force, VOLUME, KAPPA_MIN, KAPPA_MAX)

# ----------------------------------- postprocessing ----------------------------------
if args.save:
    save_designs(args, "heat_conduction", designs, compliance)
plot_frames(designs, "benchmark_heat_conduction", args.book)
