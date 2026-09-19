import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import mlhp
import numpy as np
import scipy.sparse
import torch
from tqdm import tqdm

from solvers.optimization import (
    ComplexStructuredFEM,
    DensityFilter,
    dprojection,
    projection,
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_helmholtz_adam"

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# ceiling acoustic topology optimization

# geometry & objective
LENGTHS = [18.0, 18.0]
SOURCE_CENTER = [2.0, 2.0]  # harmonic point source (bottom-left)
SOURCE_WIDTH = 0.3  # Gaussian emulation of the point source
CEILING_HEIGHT = 1.0
TARGET_CENTER = [16.0, 2.0]  # Omega_s: quiet box
TARGET_SIZE = [2.0, 2.0]

ceil_min, ceil_max = LENGTHS[1] - CEILING_HEIGHT, LENGTHS[1]
target_min = [
    TARGET_CENTER[0] - TARGET_SIZE[0] / 2,
    TARGET_CENTER[1] - TARGET_SIZE[1] / 2,
]
target_max = [
    TARGET_CENTER[0] + TARGET_SIZE[0] / 2,
    TARGET_CENTER[1] + TARGET_SIZE[1] / 2,
]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 24
SUB_VOXELS = 4
DEGREE = 2  # not sufficient TODO
QUAD_ORDER = DEGREE + 1  # integration

# physics
RHO1, RHO2 = 1.204, 2643.0  # air, aluminium
KAPPA1, KAPPA2 = 1.419e5, 6.87e10
RHO_RATIO, KAPPA_RATIO = RHO1 / RHO2, KAPPA1 / KAPPA2
FREQ = 67.43  # computed with f = lambda n, m : c/2*math.sqrt((n/L)**2 + (m/L)**2)
OMEGA = 2.0 * np.pi * FREQ / np.sqrt(KAPPA1 / RHO1)
DAMP = 0.01
SOURCE_AMP = 10.0
P0 = 20e-6  # reference pressure
RMIN = 2

# postprocessing
THRESHOLD = 0.5

# optimization (Adam)
MAX_ITER = 300
LR = 5e-2  # 1e-1
INITIAL_GUESS = 0.5  # 1.0
ETA = 0.5  # projection
BETA_MAX = 200.0  # beta-continuation
BETA_GROWTH = 1.02

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
elem_lengths = [LENGTHS[0] / nelx_e, LENGTHS[1] / nely_e]

mesh = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[nelx_e, nely_e], lengths=LENGTHS))
basis = mlhp.makeHpTensorSpace(mesh, degree=DEGREE, nfields=1)
ndof = basis.ndof()
efts = np.array(basis.locationMaps())

# --------------------------- preintegrate reference element --------------------------
# hack: differing from how helmholtz2D.py handles helmholtz: using K, M directly
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=1)

quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS, SUB_VOXELS])
order = mlhp.absoluteQuadratureOrder([QUAD_ORDER, QUAD_ORDER])
K_locals = mlhp.integratePartitionMatrices(
    basis_local,
    mlhp.poissonIntegrand(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, 0.0)),
    quadrature,
    order,
)
M_locals = mlhp.integratePartitionMatrices(
    basis_local,
    mlhp.l2DomainIntegrand(rhs=None, mass=mlhp.scalarField(2, 1.0)),
    quadrature,
    order,
)

# ------------------------------ source & target operator -----------------------------
domain = mlhp.implicitCube([0.0, 0.0], LENGTHS)
no_bc = mlhp.combineDirichletDofs([])

r2 = f"((x - {SOURCE_CENTER[0]})**2 + (y - {SOURCE_CENTER[1]})**2)"
gaussian = mlhp.scalarField(2, f"exp(-{r2} / (2 * {SOURCE_WIDTH}**2))")
m = mlhp.allocateSparseMatrix(basis, no_bc[0])
v = mlhp.allocateRhsVector(m)
mlhp.integrateOnDomain(
    basis,
    mlhp.l2DomainIntegrand(rhs=gaussian, mass=None),
    [v],
    quadrature=mlhp.spaceTreeQuadrature(domain, depth=QUAD_ORDER, epsilon=1e-8),
    dirichletDofs=no_bc,
)
force = (SOURCE_AMP * np.asarray(v)).astype(np.complex128)

target_box = mlhp.implicitCube(target_min, target_max)
mq = mlhp.allocateSparseMatrix(basis, no_bc[0])
mlhp.integrateOnDomain(
    basis,
    mlhp.l2DomainIntegrand(rhs=None, mass=mlhp.scalarField(2, 1.0)),
    [mq],
    quadrature=mlhp.spaceTreeQuadrature(target_box, depth=QUAD_ORDER, epsilon=1e-8),
    dirichletDofs=no_bc,
)
Q = scipy.sparse.csr_matrix(
    (
        np.asarray(mq.data_array),
        np.asarray(mq.indices_array),
        np.asarray(mq.indptr_array),
    ),
    shape=tuple(mq.shape),
)
Q = Q / float(Q.sum())

# --------------------------- FEM assembly & solver helpers ---------------------------
free = np.arange(ndof)
fem = ComplexStructuredFEM(efts, free, ndof, K_locals, M_locals, (NX, NY), SUB_VOXELS)

MASS_COEFF = 1j * OMEGA * DAMP + OMEGA**2
dS_local = (RHO_RATIO - 1.0) * K_locals - MASS_COEFF * (KAPPA_RATIO - 1.0) * M_locals


def build_system(zeta):
    rho_inv = 1.0 + zeta * (RHO_RATIO - 1.0)
    kappa_inv = 1.0 + zeta * (KAPPA_RATIO - 1.0)
    return fem.system(rho_inv, -MASS_COEFF * kappa_inv)


# --------------------------- density filter & Heaviside projection -------------------
density_filter = DensityFilter(RMIN, (NX, NY))

# ------------------------------------ design region ----------------------------------
voxel_y = (np.arange(NY) + 0.5) * LENGTHS[1] / NY  # y-centroid of each design-voxel row
band = np.tile((voxel_y >= ceil_min) & (voxel_y <= ceil_max), (NX, 1)).astype(float)
n_band = band.sum()
active = np.flatnonzero(band.ravel())  # design variables live only inside the band


def expand(xv):  # active band variables -> full (NX, NY) raw design (air elsewhere)
    full = np.zeros(NX * NY)
    full[active] = xv.ravel()
    return full.reshape(NX, NY)


def physical(x_tilde, beta, eta):  # projected density, pinned to air outside the band
    return projection(x_tilde, beta, eta) * band


def objective(zeta):  # mean square pressure in the target box + state solution
    lu = fem.factorize(build_system(zeta))
    p = lu.solve(force)
    return float((p.conj() @ (Q @ p)).real), p, lu


def sensitivity(zeta, x_tilde, beta, eta):  # phi and dphi/dx of one projected design
    phi, p, lu = objective(zeta)
    # adjoint S^H lam = -Q p; S is complex-symmetric so S^H = conj(S) and lu is reused
    lam = np.conj(lu.solve(-(Q @ np.conj(p))))
    # dphi/dzeta = 2 Re( lam^H dS/dzeta p ) per sub-voxel (Eq. 15)
    dzeta = 2.0 * fem.bilinear(dS_local, np.conj(lam), p).real
    return phi, density_filter.adjoint(dzeta * dprojection(x_tilde, beta, eta) * band)


# ------------------------------------ optimization -----------------------------------
n = active.size
x = torch.full((n,), INITIAL_GUESS, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=LR)

beta = 1.0
history = []

phi0 = objective(physical(density_filter(expand(x.detach().numpy())), beta, ETA))[0]
phi0 = phi0 if phi0 > 0.0 else 1.0

if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    x_tilde = density_filter(expand(x.detach().numpy()))
    rho = physical(x_tilde, beta, ETA)

    phi, grad = sensitivity(rho, x_tilde, beta, ETA)
    optimizer.zero_grad()
    x.grad = torch.from_numpy(grad.ravel()[active] / phi0)
    optimizer.step()
    with torch.no_grad():
        x.clamp_(0.0, 1.0)

    history.append(phi)
    pbar.set_postfix(
        {
            "Lp": f"{10.0 * np.log10(phi / P0**2):.1f}",
            "vol": f"{rho.sum() / n_band:.3f}",
            "beta": f"{beta:.0f}",
        }
    )

    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:04d}.jpg")
        plt.close()

    beta = min(BETA_GROWTH * beta, BETA_MAX)

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
x_int = physical(density_filter(expand(x.detach().numpy())), beta, ETA)
rho_thresh = (x_int > THRESHOLD).astype(float) * band

phi_air = objective(np.zeros((NX, NY)))[0]  # no-ceiling baseline (all air)
phi_int, _, _ = objective(x_int)
phi_thresh, p_thresh, _ = objective(rho_thresh)
sound_level = lambda p: 10.0 * np.log10(p / P0**2)
print(
    f"no ceiling   L_p {sound_level(phi_air):.1f} dB\n"
    f"intermediate L_p {sound_level(phi_int):.1f} dB  vol {x_int.sum() / n_band:.3f}\n"
    f"thresholded  L_p {sound_level(phi_thresh):.1f} dB  vol {rho_thresh.sum() / n_band:.3f}"
)

p_re, p_im = p_thresh.real, p_thresh.imag
postmesh = mlhp.domainCellMesh(domain, [DEGREE + 1] * 2)
pressure = mlhp.DataAccumulator()
mlhp.basisOutput(
    basis,
    cellmesh=postmesh,
    processors=[
        mlhp.solutionProcessor(2, mlhp.DoubleVector(p_re.tolist()), "Re"),
        mlhp.solutionProcessor(2, mlhp.DoubleVector(p_im.tolist()), "Im"),
    ],
    output=pressure,
)
re, im = np.array(pressure.data()[0]), np.array(pressure.data()[1])
spl = 10.0 * np.log10((re**2 + im**2) / P0**2 + 1e-12)
levels = np.linspace(spl.max() - 60, spl.max(), 64)  # 60 dB dynamic range

pressure_cmap = cmr.get_sub_cmap(cmr.fusion_r, 0.5, 1.0)
fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(
    pressure.triangulation(mpl=True),
    spl,
    levels=levels,
    cmap=pressure_cmap,
    extend="min",
)
design = np.ma.masked_where(rho_thresh.T < 0.5, rho_thresh.T)
ax.imshow(
    design,
    origin="lower",
    extent=[0.0, LENGTHS[0], 0.0, LENGTHS[1]],
    cmap="binary",
    vmin=0.0,
    vmax=2.5,
    zorder=2,
)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "topopt_helmholtz.pdf")
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()
