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
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from DL import sinusoidal_embedding
from solvers.optimization import DensityFilter, StructuredFEM

BASE_DIR = Path(__file__).parent
MODEL_DIR = (BASE_DIR / "../../models").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_latent_diffusion"

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# half mbb beam optimized inside the fiber-microstructure manifold of a diffusion model;

# geometry
RESOLUTION = 256
LENGTHS = [1.0, 1.0]

# discretization
SUB_VOXELS = 4
DEGREE = 3
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.6  # 0.7
RMIN = 2
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# simp penalisation continuation: ramp the exponent to sharpen the design over time
PENAL0, PENAL_INC, PENAL_MAX = 3.0, 0.02, 1.0  # TODO SIMP NOT NEEDED

# volume penalty continuation: grow the quadratic constraint weight with iterations
PENALTY0, PENALTY_INC, PENALTY_MAX = (
    1.0,
    1.0,
    500.0,
)  # TODO increase? NOT ADDITIVE BUT MULTIPLICATIVE -> SHOULD BE SLOWER IN BEGINNING BUT HIGHER IN THE END

# optimization (adam on the latent, polynomial lr decay (BETA * iter + 1) ** ALPHA)
ITERS = 200
LR = 1e-3  # a sample is far more sensitive to its starting noise than to a code
ALPHA = -0.5
BETA = 0.1
CLIP = 0.1  # gradient-norm clipping

# deterministic ddim subsequence: differentiable end to end, fewer unet evaluations
DDIM_STEPS = 25

# postprocessing
THRESHOLD = 0.5

# --------------------------- instantiate model & optimizer ---------------------------
model = torch.load(
    MODEL_DIR / "fiber_diffusion_200_256.pt2", weights_only=False, map_location=device
)
model.eval()
model.requires_grad_(False)

TIME_CHANNELS = 32  # width of the sinusoidal timestep embedding
steps = torch.linspace(model.T - 1, 0, DDIM_STEPS).long().to(device)
alpha_bars = model.alpha_bars[steps]
alpha_bars_prev = torch.cat([alpha_bars[1:], torch.ones(1, device=device)])


def denoise(x, t):
    embedding = model["time"](sinusoidal_embedding(t, TIME_CHANNELS))
    return model["head"](model["unet"](x, embedding))


# the mass of a standard normal sits on a shell of radius sqrt(D), so the latent is
# parameterized by direction alone and the radius is pinned to that shell
DIMENSION = RESOLUTION**2
latent = nn.Parameter(torch.randn(1, 1, RESOLUTION, RESOLUTION, device=device))


def noise():
    return np.sqrt(DIMENSION) * latent / latent.norm()


def step(x, alpha_bar, alpha_bar_prev, t):  # eq:ddim_step
    eps_pred = denoise(x, t)
    x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
    # clamp in the forward pass only, it saturates and would cut the gradient
    x0_pred = x0_pred + (x0_pred.clamp(-1, 1) - x0_pred).detach()
    x = torch.sqrt(alpha_bar_prev) * x0_pred
    x = x + torch.sqrt(1 - alpha_bar_prev) * eps_pred
    return x, x0_pred


def forward():
    x = noise()
    for i in range(DDIM_STEPS):
        t = steps[i].repeat(x.shape[0])
        # recompute each step in the backward pass, the activations do not fit on a gpu
        x, x0_pred = checkpoint(
            step, x, alpha_bars[i], alpha_bars_prev[i], t, use_reentrant=False
        )
    fibers = ((x0_pred + 1) / 2).clamp(0.0, 1.0)
    return 1.0 - fibers


rho_init = forward()[0, 0].detach().cpu().numpy()  # sampled starting design

optimizer = torch.optim.Adam([latent], lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda it: (BETA * it + 1) ** ALPHA
)

# -------------------------------- finite element setup -------------------------------
cells = RESOLUTION // SUB_VOXELS
elem_lengths = [length / cells for length in LENGTHS]

mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[cells, cells], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=2)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

# all elements are identical, so one reference element is preintegrated and reused
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=2)

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


# faces: 0=left, 1=right, 2=bottom, 3=top
def face_dofs(face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


symmetry = face_dofs(0, 0)  # left edge
roller = np.intersect1d(face_dofs(2, 1), face_dofs(1, 1))  # bottom-right corner
load_dof = np.intersect1d(face_dofs(3, 1), face_dofs(0, 1))  # top-left corner

fixed = np.unique(np.concatenate([symmetry, roller]))
free = np.setdiff1d(np.arange(ndof), fixed)

force = np.zeros(ndof)
force[load_dof] = LOAD
force_free = force[free]

fem = StructuredFEM(efts, free, ndof, K_locals, (RESOLUTION, RESOLUTION), SUB_VOXELS)
density_filter = DensityFilter(RMIN, (RESOLUTION, RESOLUTION))


def simp(rho):  # simp stiffness interpolation between void and solid
    return EMIN + rho**penal * (E0 - EMIN)


# ------------------------------------ optimization -----------------------------------
penal = PENAL0
penalty = PENALTY0
compliance0 = None
drift_history = [0] * ITERS
start = latent.detach().clone()

if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
tic = time.time()
pbar = tqdm(range(ITERS))
for it in pbar:
    rho_pred = forward()
    rho = rho_pred[0, 0].detach().cpu().numpy()

    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho), force_free)
    compliance = force @ u
    if compliance0 is None:
        compliance0 = compliance  # normalise the compliance sensitivity once

    # compliance sensitivity on the design grid, then the classic sensitivity filter
    dc = -penal * rho ** (penal - 1) * (E0 - EMIN) * fem.element_energy(u)
    dc = density_filter.sensitivity(rho, dc)

    # quadratic volume penalty: (mean_rho / VOLFRAC - 1) ** 2, weight grows each iter
    mean_rho = rho.mean()
    dv = 2 * (mean_rho / VOLFRAC - 1) / (VOLFRAC * RESOLUTION**2)
    sensitivity = dc / compliance0 + penalty * dv

    # the sensitivity is the incoming gradient of rho, so backpropagation through the
    # whole reverse chain turns it into a gradient on the starting noise
    optimizer.zero_grad()
    sensitivity = torch.from_numpy(sensitivity).reshape(1, 1, RESOLUTION, RESOLUTION)
    rho_pred.backward(sensitivity.to(device))

    # angle travelled on the shell, the counterpart of the rarity of the vae code
    cosine = torch.cosine_similarity(latent.detach().flatten(), start.flatten(), dim=0)
    drift_history[it] = torch.arccos(cosine.clamp(-1, 1)).item()

    torch.nn.utils.clip_grad_norm_([latent], CLIP)
    optimizer.step()
    scheduler.step()

    penal = min(penal + PENAL_INC, PENAL_MAX)
    penalty = min(penalty + PENALTY_INC, PENALTY_MAX)
    pbar.set_postfix(
        c=f"{compliance:.2e}",
        vol=f"{mean_rho:.3f}",
        p=f"{penal:.2f}",
        a=f"{drift_history[it]:.1e}",
    )

    if args.animate:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:d}.jpg")
        plt.close()

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s for {ITERS} iter")

# ----------------------------------- postprocessing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(rho_thresh), force_free)
compliance_thresh = force @ u
print(f"thresholded c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}")
print(f"latent drift {drift_history[0]:.3e} -> {drift_history[-1]:.3e} rad")

if not args.book and not args.animate:
    fig, ax = plt.subplots()
    ax.plot(drift_history, "k")
    ax.set_xlabel("iteration")
    ax.set_ylabel("latent angle")
    plt.show()

    for field in (rho_init, rho, rho_thresh):
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=150)
        ax.imshow(
            field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T
        )
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
# -------------------------------- book postprocessing --------------------------------
elif args.book:
    fields = (
        (rho_init, "topopt_latent_diffusion_init"),
        (rho, "topopt_latent_diffusion"),
        (rho_thresh, "topopt_latent_diffusion_thresh"),
    )
    for field, name in fields:
        fig, ax = plt.subplots(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=150)
        ax.imshow(
            field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_rasterized(True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf", transparent=True)
        plt.close()
