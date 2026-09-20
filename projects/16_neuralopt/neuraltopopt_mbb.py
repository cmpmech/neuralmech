import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import DCN, MLP
from solvers.optimization import DensityFilter, StructuredFEM, dsimp, simp

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/neuraltopopt_mbb"

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because the networks are small

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# half MBB beam (left edge is the symmetry plane), reparametrized by a network
# geometry
LENGTHS = [3.0, 1.0]

# discretization
N = 96
NX, NY = np.array(LENGTHS).astype(int) * N
SUB_VOXELS = 6
DEGREE = 3
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.5
RMIN = 2
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# SIMP penalisation continuation: ramp the exponent to sharpen the design over time
PENAL0, PENAL_INC, PENAL_MAX = 3.0, 0.01, 4.0

# volume penalty continuation: grow the quadratic constraint weight with iterations
PENALTY0, PENALTY_INC, PENALTY_MAX = 0.1, 0.05, 100.0

# postprocessing
THRESHOLD = 0.5

# optimization (per-ansatz lr and polynomial lr decay (BETA * iter + 1) ** ALPHA)
ANSATZ = "dcn"  # dcn, mlp or linear
MAX_ITER = 150
CLIP = 0.1
HYPERPARAMS = {
    "dcn": dict(lr=5e-3, alpha=-0.5, beta=0.2),
    "mlp": dict(lr=5e-3, alpha=-0.5, beta=0.2),
    "linear": dict(lr=5e-2, alpha=0.0, beta=0.0),
}
LR = HYPERPARAMS[ANSATZ]["lr"]
ALPHA = HYPERPARAMS[ANSATZ]["alpha"]
BETA = HYPERPARAMS[ANSATZ]["beta"]


# ----------------------------------- design ansatz -----------------------------------
class Gaussian(nn.Module):  # smooth, saturating activation; not in torch.nn
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x**2) / (2 * self.sigma**2))


if ANSATZ == "dcn":  # convolutional generator: fixed latent image -> density field
    CHANNELS = [32, 16, 8, 4, 2]
    N_UP = len(CHANNELS) - 1  # number of x2 upsamplings -> latent starts at NX/16
    assert NX % 2**N_UP == 0 and NY % 2**N_UP == 0, (
        f"design grid {[NX, NY]} must be divisible by {2**N_UP}"
    )

    KERNEL_SIZE, STRIDE, PADDING = 5, 1, 2
    SIGMA = 0.5

    pre_modules = [
        [nn.Upsample(scale_factor=2, mode="nearest"), nn.BatchNorm2d(CHANNELS[i])]
        for i in range(len(CHANNELS) - 1)
    ]
    ACTIVATIONS = [Gaussian(SIGMA) for _ in range(len(CHANNELS) - 2)]
    ACTIVATIONS += [nn.Softmax(dim=1)]

    model = DCN(
        CHANNELS,
        ACTIVATIONS,
        KERNEL_SIZE,
        STRIDE,
        PADDING,
        pre_modules=pre_modules,
        bias=True,
    ).to(device)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    last_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    nn.init.normal_(last_conv.weight, std=0.01)

    latent_size = np.array([NX, NY]) // 2**N_UP
    latent = torch.randn((1, CHANNELS[0], latent_size[0], latent_size[1]))
    latent = latent / (latent.max() - latent.min()) * 2  # span the activation's range
    latent = latent.to(device)
    params = list(model.parameters())
    forward = lambda: model(latent)[:, 0:1]  # density = first softmax channel

elif ANSATZ == "mlp":  # coordinate network: (x, y) -> density (implicit field)
    HIDDEN_LAYERS, NEURONS = 5, 128
    LAYERS = [2] + HIDDEN_LAYERS * [NEURONS] + [2]
    ACTIVATIONS = [nn.ReLU(inplace=True) for _ in range(len(LAYERS) - 2)]
    ACTIVATIONS += [nn.Softmax(dim=1)]
    normalizations = [nn.BatchNorm1d(NEURONS) for _ in range(len(LAYERS) - 2)]

    post_modules = [[nm, act] for nm, act in zip(normalizations + [None], ACTIVATIONS)]
    model = MLP(LAYERS, post_modules).to(device)
    init_weights(model, ACTIVATIONS[0])
    last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.normal_(last_linear.weight, std=0.01)
    nn.init.zeros_(last_linear.bias)

    x = torch.linspace(-1.0, 1.0, NX)
    y = torch.linspace(-1.0, 1.0, NY)
    x, y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)
    params = list(model.parameters())
    forward = lambda: model(coords)[:, 0:1].reshape(1, 1, NX, NY)

elif ANSATZ == "linear":  # no network: voxel densities are the design variables
    rho_var = nn.Parameter(torch.full((1, 1, NX, NY), VOLFRAC)).to(device)
    params = [rho_var]
    forward = lambda: rho_var

optimizer = torch.optim.Adam(params, lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda it: (BETA * it + 1) ** ALPHA
)

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
elem_lengths = [LENGTHS[0] / nelx_e, LENGTHS[1] / nely_e]

mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx_e, nely_e], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=2)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

# --------------------------- preintegrate reference element --------------------------
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=2)
ndof_e = basis_local.ndof()

material = mlhp.planeStressMaterial(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, NU))
integrand = mlhp.staticDomainIntegrand(
    mlhp.smallStrainKinematics(2), material, mlhp.vectorField(2, [0.0, 0.0])
)
quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS, SUB_VOXELS])
K_locals = mlhp.integratePartitionMatrices(
    basis_local,
    integrand,
    quadrature,
    mlhp.absoluteQuadratureOrder([QUAD_ORDER, QUAD_ORDER]),
)


# -------------------------------- boundary conditions --------------------------------
# faces: 0=left, 1=right, 2=bottom, 3=top
def face_dofs(face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


symmetry = face_dofs(0, 0)
roller = np.intersect1d(face_dofs(2, 1), face_dofs(1, 1))  # bottom-right corner
load_dof = np.intersect1d(face_dofs(3, 1), face_dofs(0, 1))  # top-left corner

fixed = np.unique(np.concatenate([symmetry, roller]))
free = np.setdiff1d(np.arange(ndof), fixed)

force = np.zeros(ndof)
force[load_dof] = LOAD
force_free = force[free]

# --------------------------- FEM assembly & solver helpers ---------------------------
fem = StructuredFEM(efts, free, ndof, K_locals, (NX, NY), SUB_VOXELS)


# ----------------------------------- density filter ----------------------------------
density_filter = DensityFilter(RMIN, (NX, NY))

# ------------------------------------ optimization -----------------------------------
penal = PENAL0
penalty = PENALTY0
compliance0 = None

if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    rho_ = forward()
    rho = rho_[0, 0].detach().cpu().numpy()

    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho, penal, EMIN, E0), force_free)
    compliance = force @ u
    if compliance0 is None:
        compliance0 = compliance

    dc = -dsimp(rho, penal, EMIN, E0) * fem.element_energy(u)
    dc = density_filter.sensitivity(rho, dc)

    mean_rho = rho.mean()
    dv = 2 * (mean_rho / VOLFRAC - 1) / (VOLFRAC * NX * NY)
    sensitivity = dc / compliance0 + penalty * dv

    optimizer.zero_grad()
    rho_.backward(torch.from_numpy(sensitivity).unsqueeze(0).unsqueeze(0).to(device))
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    optimizer.step()
    scheduler.step()
    if ANSATZ == "linear":
        rho_var.data.clamp_(0.0, 1.0)

    penal = min(penal + PENAL_INC, PENAL_MAX)
    penalty = min(penalty + PENALTY_INC, PENALTY_MAX)

    pbar.set_postfix(c=f"{compliance:.3e}", vol=f"{mean_rho:.3f}", p=f"{penal:.2f}")

    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:d}.jpg")
        plt.close()

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {MAX_ITER} iter\n"
    f"time per iter {(toc - tic) / MAX_ITER:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(rho_thresh, penal, EMIN, E0), force_free)
compliance_thresh = force @ u
print(f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}")

for field, name in ((rho, "neuraltopopt_mbb"), (rho_thresh, "neuraltopopt_mbb_thresh")):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if args.book:
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf", transparent=True)
        plt.close()
    elif not args.animate:
        plt.show()
    else:
        plt.close()

indicator_field = mlhp.scalarFieldFromVoxelData(
    mlhp.FloatVector(rho_thresh.ravel("C").astype(np.float32)),
    nvoxels=[NX, NY],
    lengths=LENGTHS,
)
processors = [
    mlhp.solutionProcessor(2, mlhp.DoubleVector(u.tolist()), "Displacement"),
    mlhp.functionProcessor(indicator_field, "Indicator"),
]
postmesh = mlhp.gridCellMesh([DEGREE + 2, DEGREE + 2])
acc = mlhp.DataAccumulator()
mlhp.basisOutput(basis, postmesh, acc, processors)

uy = np.array(acc.data()[0])[1::2]
indicator = np.array(acc.data()[1])
tri = acc.triangulation()
tri.set_mask(indicator[tri.triangles].mean(axis=1) < 0.5)

fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(tri, uy, cmap="turbo", levels=64)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "neuraltopopt_mbb_uy.pdf", transparent=True)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()
