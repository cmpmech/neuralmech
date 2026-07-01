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
from matplotlib.tri import Triangulation
from tqdm import tqdm

from solvers.optimization import MMA, StructuredFEM

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_mechanism"
DEFORM_DIR = RESULTS_DIR / "animations/animation_frames/topopt_mechanism_deform"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# compliant force inverter: an input push at the left produces an output pull to the
# right that moves in the opposite direction. The bottom edge is the symmetry plane,
# so only the upper half of the square design domain is modelled.

# geometry
LENGTHS = [2.0, 1.0]

# discretization
NX, NY = np.array(LENGTHS).astype(int) * 200  # 100
SUB_VOXELS = 4
DEGREE = 2
QUAD_ORDER = DEGREE + 1  # integration

# physics
VOLFRAC = 0.3
RMIN = 3  # 3  # 5  # TODO quite large
E0, EMIN, NU = 1.0, 1e-9, 0.3
F_IN = 1.0  # input actuation force (+x at the input port)
K_IN = 1.0  # input-port spring (stiff actuator)
K_OUT = 0.01  # output-port spring (soft workpiece resistance)
PORT = 12  # passive solid patch (design voxels) anchoring each port to the structure

# postprocessing
THRESHOLD = 0.5

# optimization (MMA) with penalization continuation: a mechanism cannot be grown
# from a fully penalized gray start (the springs dominate the near-void stiffness and
# the design just dissolves), so SIMP starts near-linear and stiffens over the run
MAX_ITER = 200
CHANGE_TOL = 0.005
MMA_MOVE = 0.1  # MMA step move limit
PENAL_MIN, PENAL_MAX = 1.0, 3.0
CONT_STEP = 15  # iterations between penalization increments

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


symmetry = face_dofs(2, 1)  # bottom edge: u_y = 0 on the symmetry plane
support = np.union1d(  # top-left corner clamped in both directions
    np.intersect1d(face_dofs(0, 0), face_dofs(3, 0)),
    np.intersect1d(face_dofs(0, 1), face_dofs(3, 1)),
)
input_dof = np.intersect1d(face_dofs(0, 0), face_dofs(2, 0))[0]  # bottom-left x
output_dof = np.intersect1d(face_dofs(1, 0), face_dofs(2, 0))[0]  # bottom-right x

fixed = np.unique(np.concatenate([symmetry, support]))
free = np.setdiff1d(np.arange(ndof), fixed)  # all non-fixed dofs
dof_map = np.full(ndof, -1)
dof_map[free] = np.arange(free.size)

# input load, output measurement vector, and the two nodal springs
force = np.zeros(ndof)
force[input_dof] = F_IN
force_free = force[free]

probe = np.zeros(ndof)
probe[output_dof] = 1.0  # u_out = probe^T u, the quantity we drive negative
probe_free = probe[free]

spring_free = np.zeros(free.size)
spring_free[dof_map[input_dof]] += K_IN
spring_free[dof_map[output_dof]] += K_OUT

# passive solid patches at both ports: without an anchor the optimizer escapes the
# objective by simply tearing the input/output node loose (output then never moves)
passive = np.zeros((NX, NY), dtype=bool)
passive[:PORT, :PORT] = True  # input port (bottom-left)
passive[NX - PORT :, :PORT] = True  # output port (bottom-right)

# --------------------------- FEM assembly & solver helpers ---------------------------
fem = StructuredFEM(efts, free, ndof, K_locals, (NX, NY), SUB_VOXELS)


def simp(rho, penal):  # SIMP stiffness interpolation between void and solid
    return EMIN + rho**penal * (E0 - EMIN)


# ----------------------------------- density filter ----------------------------------
ceil_r = int(np.ceil(RMIN))
ky, kx = np.meshgrid(np.arange(-ceil_r, ceil_r + 1), np.arange(-ceil_r, ceil_r + 1))
kernel = np.maximum(0.0, RMIN - np.sqrt(kx**2 + ky**2))
Hs = scipy.ndimage.convolve(np.ones((NX, NY)), kernel, mode="constant", cval=0.0)


def density_filter(x):  # conic smoothing of the raw design
    return scipy.ndimage.convolve(x, kernel, mode="constant", cval=0.0) / Hs


def filter_adjoint(g):  # transpose of density_filter, for the chain rule
    return scipy.ndimage.convolve(g / Hs, kernel, mode="constant", cval=0.0)


# ------------------------------------ optimization -----------------------------------
n = NX * NY
mma = MMA(n, move=MMA_MOVE)
xval = np.full((n, 1), VOLFRAC)

penal = PENAL_MIN
scale = None
history = []

ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None
tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    x = xval.reshape(NX, NY).copy()
    x[passive] = 1.0  # pin the port patches solid before filtering
    rho = density_filter(x)

    # state solve and adjoint share K(rho) + springs; probe is the adjoint load
    u = np.zeros(ndof)
    u[free] = fem.solve(simp(rho, penal), force_free, spring_diag=spring_free)
    lam = np.zeros(ndof)
    lam[free] = fem.solve(simp(rho, penal), probe_free, spring_diag=spring_free)
    u_out = probe @ u

    # d(u_out)/d(rho) = -lam^T (dK/drho) u, mapped back through the filter
    dudout = -penal * rho ** (penal - 1) * (E0 - EMIN) * fem.bilinear(K_locals, lam, u)
    dc = filter_adjoint(dudout)

    vol = rho.mean()
    dvol = filter_adjoint(np.full((NX, NY), 1.0 / n))

    scale = abs(u_out) if scale is None else scale
    f0val = u_out / scale  # minimize output displacement -> output moves opposite
    df0dx = dc.ravel() / scale
    fval = vol / VOLFRAC - 1.0
    dfdx = dvol.ravel() / VOLFRAC

    xnew = mma.step(xval, f0val, df0dx, fval, dfdx)
    change = float(np.abs(xnew - xval).max())
    xval = xnew
    history.append(u_out)
    pbar.set_postfix(
        {
            "u_out": f"{u_out:.3e}",
            "vol": f"{vol:.3f}",
            "p": f"{penal:.2f}",
            "ch": f"{change:.2e}",
        }
    )
    if args.animate:
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(rho.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(ANIMATION_DIR / f"frame_{it:d}.jpg")
        plt.close()

    if penal < PENAL_MAX and it > 0 and it % CONT_STEP == 0:
        penal = min(penal + 0.25, PENAL_MAX)
    elif penal >= PENAL_MAX and change < CHANGE_TOL:
        break

toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ----------------------------------- postprocessing ----------------------------------
x = xval.reshape(NX, NY).copy()
x[passive] = 1.0
rho = density_filter(x)
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = fem.solve(simp(rho, penal), force_free, spring_diag=spring_free)
u_out_int = probe @ u
u[free] = fem.solve(simp(rho_thresh, penal), force_free, spring_diag=spring_free)
u_out_thresh = probe @ u
u_in_thresh = force @ u / F_IN  # input-port displacement (work-conjugate to F_IN)
print(
    f"intermediate u_out {u_out_int:.3e}  vol {rho.mean():.3f}\n"
    f"thresholded  u_out {u_out_thresh:.3e}  u_in {u_in_thresh:.3e}\n"
    f"geometric advantage {-u_out_thresh / u_in_thresh:.3f}  vol {rho_thresh.mean():.3f}"
)

if not args.book and not args.animate:
    for field in (rho, rho_thresh):
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.imshow(
            field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

# x-displacement on the thresholded structure (void left transparent), showing how the
# input port travels +x while the output port is inverted to -x
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

ux = np.array(acc.data()[0])[0::2]
uy = np.array(acc.data()[0])[1::2]
indicator = np.array(acc.data()[1])
tri = acc.triangulation()
mask = indicator[tri.triangles].mean(axis=1) < 0.5
tri.set_mask(mask)

fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(tri, ux, cmap="turbo", levels=64)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)  # vectorized pdf too large at this mesh density
fig.tight_layout(pad=0)
if args.book:
    plt.savefig(RGB_PDF_DIR / "topopt_mechanism_ux.pdf", transparent=True)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()

# ------------------- deformation animation (linear load 0 -> full scale) -------------
# the response is linear, so a growing load just scales u and its x-contour together
if args.animate:
    DEFORM_DIR.mkdir(parents=True, exist_ok=True)
    N_FRAMES = 60
    solid = indicator > 0.5  # ignore the near-rigid-body drift of void nodes
    scale = 0.2 * LENGTHS[0] / np.sqrt(ux**2 + uy**2)[solid].max()
    levels = np.linspace((scale * ux)[solid].min(), (scale * ux)[solid].max(), 64)
    # bound both the undeformed (frame 0) and full-load configurations, plus a margin
    xs = np.concatenate([tri.x[solid], (tri.x + scale * ux)[solid]])
    ys = np.concatenate([tri.y[solid], (tri.y + scale * uy)[solid]])
    pad = 0.03 * LENGTHS[0]
    xlim = [xs.min() - pad, xs.max() + pad]
    ylim = [ys.min() - pad, ys.max() + pad]
    for f, frac in enumerate(np.linspace(0.0, 1.0, N_FRAMES)):
        s = frac * scale
        tri_f = Triangulation(tri.x + s * ux, tri.y + s * uy, tri.triangles)
        tri_f.set_mask(mask)
        fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
        ax.tricontourf(tri_f, s * ux, levels=levels, cmap="turbo", extend="both")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.savefig(DEFORM_DIR / f"frame_{f:d}.jpg")
        plt.close()
