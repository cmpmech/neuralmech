import argparse
import os

import cmasher as cmr

# small system: single-threaded BLAS beats multithreaded spawn overhead
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import mmapy
import numpy as np
import scipy.ndimage
import scipy.sparse
from tqdm import tqdm

from solvers.optimization import ComplexStructuredFEM

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_helmholtz"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# ceiling acoustic topology optimization (Herrmann et al. 2024, Fig. 1 / Table 1):
# distribute aluminium in a ceiling band to suppress the sound a harmonic point source
# radiates into a target box in the opposite (bottom-right) corner of the room

# geometry (metres)
LENGTHS = [18.0, 9.0]
SOURCE_CENTER = [2.0, 2.0]  # harmonic point source (bottom-left)
SOURCE_WIDTH = 0.3  # Gaussian regularisation of the point source
CEILING_HEIGHT = 2.0  # h_c: ceiling design band occupies the top of the domain
TARGET_CENTER = [16.0, 2.0]  # Omega_s: quiet box (bottom-right)
TARGET_SIZE = [2.0, 2.0]
BAND_LO, BAND_HI = LENGTHS[1] - CEILING_HEIGHT, LENGTHS[1]
TARGET_LO = [
    TARGET_CENTER[0] - TARGET_SIZE[0] / 2,
    TARGET_CENTER[1] - TARGET_SIZE[1] / 2,
]
TARGET_HI = [
    TARGET_CENTER[0] + TARGET_SIZE[0] / 2,
    TARGET_CENTER[1] + TARGET_SIZE[1] / 2,
]

# discretization
NX, NY = 144, 72  # 432, 216  # 144, 72  # 432, 216
SUB_VOXELS = 4
DEGREE = 2
QUAD_ORDER = DEGREE + 1

# physics: linear interpolation of inverse mass density & bulk modulus (air <-> aluminium)
RHO1, RHO2 = 1.204, 2643.0  # mass density [kg/m^3]: air, aluminium
KAPPA1, KAPPA2 = 1.419e5, 6.87e10  # bulk modulus [N/m^2]: air, aluminium
RHO_RATIO, KAPPA_RATIO = RHO1 / RHO2, KAPPA1 / KAPPA2
FREQ = 69.43  # excitation frequency [Hz] (a domain resonance)
OMEGA = 2.0 * np.pi * FREQ / np.sqrt(KAPPA1 / RHO1)  # normalised wavenumber omega-tilde
DAMP = 0.01  # mass-proportional damping coefficient eta_d
SOURCE_AMP = 10.0  # source amplitude s-hat [Pa/m^2]
P0 = 2e-6  # reference pressure for the sound pressure level [Pa]
VOLFRAC = 0.5  # NOT NEEDED  # 0  # 1  # 0  # 0.4
RMIN = 2

# post-processing
THRESHOLD = 0.5

# optimization with beta-continuation
MAX_ITER = 300  # 5  # 300  # 5  # 300
CHANGE_TOL = 0.01
MMA_MOVE = 0.2
ETA_E, ETA_I, ETA_D = 0.6, 0.5, 0.4  # eroded / intermediate / dilated thresholds
BETA_MAX = 32.0
CONT_STEP = 25

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
# stiffness K_e = int(grad N . grad N) and mass M_e = int(N N), scaled independently
mesh_local = mlhp.makeRefinedGrid(mlhp.makeGrid(ncells=[1, 1], lengths=elem_lengths))
basis_local = mlhp.makeHpTensorSpace(mesh_local, degree=DEGREE, nfields=1)

quadrature = mlhp.gridQuadrature(nsubcells=[SUB_VOXELS, SUB_VOXELS])
order = mlhp.absoluteQuadratureOrder([QUAD_ORDER, 2])
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

# consistent load vector of the real point source: f = int(g N)
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

# target mass matrix Q (objective is the mean square sound pressure over the box)
target_box = mlhp.implicitCube(TARGET_LO, TARGET_HI)
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
Q = Q / float(Q.sum())  # normalise so the objective is the mean square pressure

# --------------------------- FEM assembly & solver helpers ---------------------------
# homogeneous Neumann everywhere, so every dof is free; the system is assembled natively
# as an N x N complex matrix (not a 2N x 2N real block) and its LU is reused for the
# adjoint since S is complex-symmetric (S^H = conj(S))
free = np.arange(ndof)
fem = ComplexStructuredFEM(efts, free, ndof, K_locals, M_locals, (NX, NY), SUB_VOXELS)

# complex system S = rho^-1 K - (i omega eta_d + omega^2) kappa^-1 M with rho^-1 and
# kappa^-1 linearly interpolated in the design. dS/dzeta is constant, so it is
# preintegrated once and contracted per voxel for the adjoint sensitivity
MASS_COEFF = 1j * OMEGA * DAMP + OMEGA**2
dS_local = (RHO_RATIO - 1.0) * K_locals - MASS_COEFF * (KAPPA_RATIO - 1.0) * M_locals


def build_system(zeta):  # complex Helmholtz system matrix
    rho_inv = 1.0 + zeta * (RHO_RATIO - 1.0)  # rho-tilde^-1 (stiffness coefficient)
    kappa_inv = 1.0 + zeta * (KAPPA_RATIO - 1.0)  # kappa-tilde^-1 (mass coefficient)
    return fem.system(rho_inv, -MASS_COEFF * kappa_inv)


# --------------------------- density filter & Heaviside projection -------------------
ceil_r = int(np.ceil(RMIN))
ky, kx = np.meshgrid(np.arange(-ceil_r, ceil_r + 1), np.arange(-ceil_r, ceil_r + 1))
kernel = np.maximum(0.0, RMIN - np.sqrt(kx**2 + ky**2))
Hs = scipy.ndimage.convolve(np.ones((NX, NY)), kernel, mode="constant", cval=0.0)


def density_filter(x):  # conic smoothing of the raw design
    return scipy.ndimage.convolve(x, kernel, mode="constant", cval=0.0) / Hs


def filter_adjoint(g):  # transpose of density_filter, for the chain rule
    return scipy.ndimage.convolve(g / Hs, kernel, mode="constant", cval=0.0)


def projection(x_tilde, beta, eta):  # smoothed Heaviside about threshold eta
    a, b = np.tanh(beta * eta), np.tanh(beta * (1.0 - eta))
    return (a + np.tanh(beta * (x_tilde - eta))) / (a + b)


def dprojection(x_tilde, beta, eta):  # d(projection)/d(x_tilde)
    a, b = np.tanh(beta * eta), np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x_tilde - eta)) ** 2) / (a + b)


# ------------------------------------ design region ----------------------------------
# material is confined to the ceiling band; everything else is passive air. the design
# lives on the (NX, NY) voxel grid, so the band is built from voxel centroids
voxel_y = (np.arange(NY) + 0.5) * LENGTHS[1] / NY  # y-centroid of each design-voxel row
band = np.tile((voxel_y >= BAND_LO) & (voxel_y <= BAND_HI), (NX, 1)).astype(float)
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
    return phi, filter_adjoint(dzeta * dprojection(x_tilde, beta, eta) * band)


# ------------------------------------ optimization -----------------------------------
n = active.size
xval = np.full((n, 1), VOLFRAC)
xold1, xold2 = xval.copy(), xval.copy()
low, upp = np.zeros((n, 1)), np.ones((n, 1))
xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
a0, a_mma, c_mma, d_mma = 1.0, np.zeros((1, 1)), 1e3 * np.ones((1, 1)), np.zeros((1, 1))

beta = 1.0
vfrac_d = VOLFRAC  # dilated bound, adapted so the intermediate design hits VOLFRAC
phi_ref = None
history = []

ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    x_tilde = density_filter(expand(xval))
    x_e = physical(x_tilde, beta, ETA_E)
    x_i = physical(x_tilde, beta, ETA_I)
    x_d = physical(x_tilde, beta, ETA_D)

    # robust min-max: drive down the WORST of the eroded / intermediate / dilated
    # designs (unlike compliance, a thinner acoustic barrier is not always the worst)
    phi_e, dc_e = sensitivity(x_e, x_tilde, beta, ETA_E)
    phi_i, dc_i = sensitivity(x_i, x_tilde, beta, ETA_I)
    phi_d, dc_d = sensitivity(x_d, x_tilde, beta, ETA_D)
    worst = int(np.argmax([phi_e, phi_i, phi_d]))
    phi = [phi_e, phi_i, phi_d][worst]
    dc = [dc_e, dc_i, dc_d][worst]

    # volume constraint on the dilated design; bound rescaled so x_i hits VOLFRAC
    vfrac_d *= VOLFRAC / (x_i.sum() / n_band)
    dvol = filter_adjoint(dprojection(x_tilde, beta, ETA_D) * band / n_band)

    phi_ref = phi if phi_ref is None else phi_ref
    f0val = phi / phi_ref
    df0dx = (dc.ravel()[active] / phi_ref).reshape(n, 1)
    fval = np.array([[x_d.sum() / n_band / vfrac_d - 1.0]])
    dfdx = (dvol.ravel()[active] / vfrac_d).reshape(1, n)

    xmma, _, _, _, _, _, _, _, _, low, upp = mmapy.mmasub(
        1,
        n,
        it + 1,
        xval,
        xmin,
        xmax,
        xold1,
        xold2,
        f0val,
        df0dx,
        fval,
        dfdx,
        low,
        upp,
        a0,
        a_mma,
        c_mma,
        d_mma,
        move=MMA_MOVE,
    )
    xold2, xold1 = xold1, xval
    change = np.abs(xmma - xval).max()
    xval = xmma
    history.append(phi)
    pbar.set_postfix(
        {
            "Lp": f"{10.0 * np.log10(phi / P0**2):.1f}",
            "vol": f"{x_i.sum() / n_band:.3f}",
            "beta": f"{beta:.0f}",
            "ch": f"{change:.2e}",
        }
    )

    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(x_i.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:04d}.jpg")
        plt.close()

    # beta-continuation: sharpen the projection, then stop once converged at full beta
    if beta < BETA_MAX and it > 0 and it % CONT_STEP == 0:
        beta = min(2.0 * beta, BETA_MAX)
    elif beta >= BETA_MAX and change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ---------------------------------- post-processing ----------------------------------
x_int = physical(density_filter(expand(xval)), beta, ETA_I)
rho_thresh = (x_int > THRESHOLD).astype(float) * band

phi_air = objective(np.zeros((NX, NY)))[0]  # no-ceiling baseline (all air)
phi_int, x, _ = objective(x_int)
phi_thresh, _, _ = objective(rho_thresh)
sound_level = lambda p: 10.0 * np.log10(p / P0**2)
print(
    f"no ceiling   L_p {sound_level(phi_air):.1f} dB\n"
    f"intermediate L_p {sound_level(phi_int):.1f} dB  vol {x_int.sum() / n_band:.3f}\n"
    f"thresholded  L_p {sound_level(phi_thresh):.1f} dB  vol {rho_thresh.sum() / n_band:.3f}"
)

# sound pressure level field of the intermediate design: L_p = 10 log10(|p|^2 / p0^2)
p_re, p_im = x.real, x.imag
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
spl = 10.0 * np.log10((re**2 + im**2) / P0**2 + 1e-12)  # eps avoids log(0) at nodes
levels = np.linspace(spl.max() - 80, spl.max(), 64)  # 80 dB dynamic range

fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(
    pressure.triangulation(mpl=True),
    spl,
    levels=levels,
    cmap="hot_r",
    # cmap=cmr.fall_r,
    # cmap=cmr.apple_r,
    extend="min",
)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        RESULTS_DIR / "topopt_helmholtz_field.pdf", bbox_inches="tight", pad_inches=0
    )
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()

for field, name in (
    (x_int, "topopt_helmholtz"),
    (rho_thresh, "topopt_helmholtz_thresh"),
):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    if args.book:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / f"{name}.png")
        plt.close()
    elif not args.animate:
        plt.show()
    else:
        plt.close()
