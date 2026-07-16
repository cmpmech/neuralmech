import argparse
import os

# small system: single-threaded CHOLMOD/BLAS beats multithreaded spawn overhead
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.ndimage
import torch
from torch import nn
from tqdm import tqdm

from solvers.optimization import StructuredFEM

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_mbb_fiber"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# half MBB beam (left edge is the symmetry plane)
# geometry -- square domain sized to match the fiber VAE's input images
# (see projects/15_anomaly/fiber_vae_spatial_train.py: domain_size)
DOMAIN_SIZE = 256  # pixels per side, must match the VAE's domain_size
LENGTHS = [1.0, 1.0]

# discretization
NX, NY = DOMAIN_SIZE, DOMAIN_SIZE
SUB_VOXELS = 4
DEGREE = 3
QUAD_ORDER = DEGREE + 1  # integration

# physics
VOLFRAC = 0.5
PENAL = 3.0
RMIN = 2
E0, EMIN, NU = 1.0, 1e-9, 0.3
LOAD = -1.0

# postprocessing
THRESHOLD = 0.5

# optimization (optimality criterion)
MOVE = 0.2
DAMPING = 0.5
MAX_ITER = 500
CHANGE_TOL = 0.01
RHO_INIT_CLIP = 0.05  # keep the VAE-seeded rho off exact 0/1 (zero-sensitivity trap)

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
    basis_local, integrand, quadrature, mlhp.absoluteQuadratureOrder([QUAD_ORDER, 2])
)


# -------------------------------- boundary conditions --------------------------------
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
free = np.setdiff1d(np.arange(ndof), fixed)  # all none fixed dofs

force = np.zeros(ndof)
force[load_dof] = LOAD
force_free = force[free]

# --------------------------- FEM assembly & solver helpers ---------------------------
fem = StructuredFEM(efts, free, ndof, K_locals, (NX, NY), SUB_VOXELS)


def simp(rho):  # SIMP stiffness interpolation between void and solid
    return EMIN + rho**PENAL * (E0 - EMIN)


# ----------------------------------- density filter ----------------------------------
ceil_r = int(np.ceil(RMIN))
ky, kx = np.meshgrid(np.arange(-ceil_r, ceil_r + 1), np.arange(-ceil_r, ceil_r + 1))
kernel = np.maximum(0.0, RMIN - np.sqrt(kx**2 + ky**2))
Hs = scipy.ndimage.convolve(np.ones((NX, NY)), kernel, mode="constant", cval=0.0)


def filter_sensitivity(rho, dc):
    num = scipy.ndimage.convolve(rho * dc, kernel, mode="constant", cval=0.0)
    return num / (np.maximum(rho, 1e-3) * Hs)


# ----------------------------- generative initialization -----------------------------
# reproduces the fiber VAE's classes so torch.load can unpickle the checkpoint (it was
# saved from "__main__" of projects/15_anomaly/fiber_vae_spatial_train.py)
class Standardizer(nn.Module):
    def __init__(self, X, dim=0):
        super().__init__()
        self.register_buffer("x_mean", X.mean(dim=dim, keepdim=True))
        self.register_buffer("x_std", X.std(dim=dim, keepdim=True).clamp_min(1e-8))

    def __call__(self, x):
        return (x - self.x_mean) / self.x_std


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
            activation if activation is not None else nn.PReLU(init=0.2),
        )

    def forward(self, x):
        return self.block(x)


class SpatialVAE(nn.Module):
    def __init__(self, encoder, mean_layer, logvar_layer, decoder):
        super().__init__()
        self.encoder = encoder
        self.mean_layer = mean_layer
        self.logvar_layer = logvar_layer
        self.decoder = decoder

    def encode(self, x):
        features = self.encoder(x)
        return self.mean_layer(features), self.logvar_layer(features)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, mean, logvar


vae_depth, vae_latent_channels, vae_beta = 3, 16, 1e-5
vae_model_path = (
    BASE_DIR
    / f"../../models/fiber_vae_spatial_best_d{vae_depth}_{vae_latent_channels}_{vae_beta}_{DOMAIN_SIZE}.pt2"
)
vae = torch.load(vae_model_path, weights_only=False, map_location=device)
vae.eval()

latent_size = DOMAIN_SIZE // 2**vae_depth
with torch.no_grad():
    z = torch.randn(1, vae_latent_channels, latent_size, latent_size, device=device)
    rho_init = torch.sigmoid(vae.decode(z))[0, 0].cpu().numpy().astype(np.float64)
rho_init = np.clip(rho_init, RHO_INIT_CLIP, 1.0 - RHO_INIT_CLIP)

# ------------------------------------ optimization -----------------------------------
rho = rho_init


# rho = rho / np.mean(rho) * VOLFRAC

# print(np.sum(rho))
# print(np.mean(rho))

rho *= 0
rho += 1
rho *= VOLFRAC


history = []

ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for iter in pbar:
    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho), force_free)
    compliance = force @ u

    # compliance sensitivity, mapped back to the design grid
    dc = -PENAL * rho ** (PENAL - 1) * (E0 - EMIN) * fem.element_energy(u)
    dc = filter_sensitivity(rho, dc)

    # optimality criterion update with bisection on the volume multiplier
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.maximum(
            0.0,
            np.maximum(
                rho - MOVE,
                np.minimum(
                    1.0,
                    np.minimum(
                        rho + MOVE, rho * (np.maximum(0.0, -dc) / lmid) ** DAMPING
                    ),
                ),
            ),
        )
        if rho_new.mean() > VOLFRAC:
            l1 = lmid
        else:
            l2 = lmid

    change = np.abs(rho_new - rho).max()
    rho = rho_new
    history.append(compliance)
    pbar.set_postfix({"c": f"{compliance:.3e}", "change": f"{change:.2e}"})

    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{iter:d}.jpg")
        plt.close()

    if change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {iter} iter\n"
    f"time per iter {(toc - tic) / iter:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(rho_thresh), force_free)
compliance_thresh = force @ u
print(f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}")

if not args.book and not args.animate:
    for field, name in (
        (rho, "topopt_mbb_fiber"),
        (rho_thresh, "topopt_mbb_fiber_thresh"),
    ):
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(
            field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

# y-displacement evaluated on the thresholded structure (void left transparent)
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
ax.set_rasterized(True)  # vectorized pdf too large at this mesh density
fig.tight_layout(pad=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "topopt_mbb_fiber_uy.pdf", transparent=True)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()
