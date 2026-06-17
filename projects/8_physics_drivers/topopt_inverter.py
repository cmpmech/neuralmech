import argparse
import os

# small system: single-threaded CHOLMOD/BLAS beats multithreaded spawn overhead
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cvxopt
import cvxopt.cholmod
import matplotlib.pyplot as plt
import mlhp
import nlopt
import numpy as np
import scipy.ndimage
import scipy.sparse
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/topopt_inverter"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# force inverter: an input push (+x) produces an output displacement in -x.
# top half of a square is modelled, the bottom edge is the symmetry plane.

# geometry
LENGTHS = [1.0, 0.5]

# discretization
NX, NY = 200, 100  # 2:1 aspect matching LENGTHS
SUB_VOXELS = 2
DEGREE = 1
QUAD_ORDER = DEGREE + 1

# physics
VOLFRAC = 0.3
PENAL = 3.0
PENAL_MIN = 1.0  # penalization is ramped 1 -> PENAL so the inverter basin is reachable
RMIN = 3
E0, EMIN, NU = 1.0, 1e-9, 0.3
F_IN = 1.0  # input load (+x at the input port)
K_IN, K_OUT = 0.1, 0.1  # actuator / workpiece spring stiffness at the ports

# post-processing
THRESHOLD = 0.5

# robust min-max optimization with beta- and penalization-continuation
MAX_ITER = 600
CHANGE_TOL = 0.01
ETA_E, ETA_I, ETA_D = 0.6, 0.5, 0.4  # eroded / intermediate / dilated thresholds
BETA_MAX = 16.0
CONT_STEP = 30

# ---------------------------------------- mesh ---------------------------------------
assert NX % SUB_VOXELS == 0 and NY % SUB_VOXELS == 0, (
    f"design grid {[NX, NY]} must be divisible by SUB_VOXELS={SUB_VOXELS}"
)
nelx_e, nely_e = NX // SUB_VOXELS, NY // SUB_VOXELS
N_elems = nelx_e * nely_e
n_sub = SUB_VOXELS**2
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
# faces: 0=left, 1=right, 2=bottom (symmetry), 3=top
def face_dofs(face, ifield):
    bc = mlhp.integrateDirichletDofs(
        mlhp.scalarField(2, 0.0), basis, [face], ifield=ifield
    )
    return np.array(mlhp.combineDirichletDofs([bc])[0])


symmetry = face_dofs(2, 1)  # bottom edge: u_y = 0
support = np.concatenate(  # top-left corner: clamped in both directions
    [np.intersect1d(face_dofs(0, i), face_dofs(3, i)) for i in (0, 1)]
)
input_dof = np.intersect1d(face_dofs(0, 0), face_dofs(2, 0)).item()  # bottom-left, x
output_dof = np.intersect1d(face_dofs(1, 0), face_dofs(2, 0)).item()  # bottom-right, x

fixed = np.unique(np.concatenate([symmetry, support]))
free = np.setdiff1d(np.arange(ndof), fixed)  # all non-fixed dofs

force = np.zeros(ndof)
force[input_dof] = F_IN
force_free = force[free]

L = np.zeros(ndof)  # dummy unit load at the output port for the adjoint solve
L[output_dof] = 1.0
L_free = L[free]

# --------------------------- FEM assembly & solver helpers ---------------------------
# every element contributes ndof_e^2 entries to the same (iK, jK) locs of K each iter
iK = np.repeat(efts, ndof_e, axis=1).ravel()
jK = np.tile(efts, (1, ndof_e)).ravel()


def grid_to_elements(field):  # (NX, NY) -> (N_elems, n_sub)
    return (
        field.reshape(nelx_e, SUB_VOXELS, nely_e, SUB_VOXELS)
        .transpose(0, 2, 1, 3)
        .reshape(N_elems, n_sub)
    )


def elements_to_grid(field):  # (N_elems, n_sub) -> (NX, NY)
    return (
        field.reshape(nelx_e, nely_e, SUB_VOXELS, SUB_VOXELS)
        .transpose(0, 2, 1, 3)
        .reshape(NX, NY)
    )


# see topopt_mbb.py for explanation; the input/output springs are injected onto the
# diagonal data entries (which already exist in the free-free sparsity pattern)
def build_assemble_K_free():
    dof_map = np.full(ndof, -1)
    dof_map[free] = np.arange(free.size)
    keep = (dof_map[iK] >= 0) & (dof_map[jK] >= 0)  # entries with both dofs free
    ri, rj = dof_map[iK[keep]], dof_map[jK[keep]]
    order = np.lexsort((ri, rj))  # column-major order expected by CSC
    data_idx = np.flatnonzero(keep)[order]  # gather positions into K_e.ravel()
    ri, rj = ri[order], rj[order]
    first = np.empty(ri.size, dtype=bool)
    first[0] = True
    first[1:] = (ri[1:] != ri[:-1]) | (rj[1:] != rj[:-1])
    seg = np.flatnonzero(first)  # duplicate (row, col) group boundaries
    indices = ri[first].astype(np.int32)
    indptr = np.concatenate(
        [[0], np.cumsum(np.bincount(rj[first], minlength=free.size))]
    ).astype(np.int32)

    # data-array positions of the spring diagonals
    def diag_pos(j):
        col = slice(indptr[j], indptr[j + 1])
        return indptr[j] + np.searchsorted(indices[col], j)

    spring_pos = np.array([diag_pos(dof_map[input_dof]), diag_pos(dof_map[output_dof])])
    spring_val = np.array([K_IN, K_OUT])

    def assemble_K_free(rho_field):  # penalised stiffness restricted to the free dofs
        E_e = EMIN + grid_to_elements(rho_field) ** PENAL * (E0 - EMIN)
        K_e = np.einsum("es,sij->eij", E_e, K_locals, optimize=True)
        data = np.add.reduceat(K_e.ravel()[data_idx], seg)
        data[spring_pos] += spring_val
        return scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=(free.size, free.size)
        )

    return assemble_K_free


assemble_K_free = build_assemble_K_free()


# for CHOLMOD: the SPD system's sparsity is factored symbolically once
K_free = assemble_K_free(np.full((NX, NY), VOLFRAC))
A = cvxopt.spmatrix(
    cvxopt.matrix(K_free.data),
    cvxopt.matrix(K_free.indices.tolist()),
    cvxopt.matrix(np.repeat(np.arange(free.size), np.diff(K_free.indptr)).tolist()),
    (free.size, free.size),
)
factor = cvxopt.cholmod.symbolic(A)


def factorize(rho_field):  # numeric factorization of K(rho) for reuse across rhs
    A.V = cvxopt.matrix(assemble_K_free(rho_field).data)
    cvxopt.cholmod.numeric(A, factor)


def solve(rhs_free):  # solve the factored system for one right-hand side
    b = cvxopt.matrix(rhs_free)
    cvxopt.cholmod.solve(factor, b)
    return np.array(b).ravel()


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


# ------------------------------------ optimization -----------------------------------
# robust min-max: the worst output displacement over the eroded / intermediate / dilated
# designs is minimized, which suppresses one-node hinges (the eroded design must invert
# too) and -- unlike the single design -- keeps the structure from starving toward the
# disconnected u_out = 0 minimum. epigraph form: min t  s.t.  u_out_eta <= t for each eta.
# nlopt's compiled MMA drives it; beta- and penalization-continuation run as staged calls.
n = NX * NY
nvar = n + 1  # design variables + the epigraph (worst-case) variable t
ETAS = (ETA_E, ETA_I, ETA_D)
state = {"beta": 1.0, "p": PENAL_MIN, "scale": None, "vfrac_d": VOLFRAC, "it": 0, "uo": {}}
history = []
ANIMATION_DIR.mkdir(parents=True, exist_ok=True) if args.animate else None


def epigraph(z, grad):  # minimize the worst-case (largest) output displacement t
    if grad.size > 0:
        grad[:] = 0.0
        grad[n] = 1.0
    state["it"] += 1
    pbar.update(1)
    if state["uo"]:
        history.append(state["uo"].get(ETA_I, 0.0))
        pbar.set_postfix(
            {"uo_e": f"{state['uo'].get(ETA_E, 0):.2e}",
             "uo_i": f"{state['uo'].get(ETA_I, 0):.2e}",
             "beta": f"{state['beta']:.0f}", "p": f"{state['p']:.1f}"}
        )
    return z[n]


# the non-self-adjoint mechanism objective: each realization factors K once and solves it
# twice (state field u, adjoint field lambda) before forming the adjoint sensitivity
def make_performance(eta):  # u_out of one projection realization, constrained <= t
    def performance(z, grad):
        beta, p = state["beta"], state["p"]
        x_tilde = density_filter(z[:n].reshape(NX, NY))
        x_r = projection(x_tilde, beta, eta)
        factorize(EMIN + x_r**p * (E0 - EMIN))
        u = np.zeros(ndof)
        lam = np.zeros(ndof)
        u[free] = solve(force_free)
        lam[free] = solve(L_free)
        u_out = L @ u
        ce = elements_to_grid(np.einsum("ei,sij,ej->es", lam[efts], K_locals, u[efts], optimize=True))
        dc = -p * x_r ** (p - 1) * (E0 - EMIN) * ce
        dc = filter_adjoint(dc * dprojection(x_tilde, beta, eta))
        # normalize by the (positive) input compliance, not u_out which straddles zero
        if state["scale"] is None:
            state["scale"] = max(force @ u, 1e-12)
        if grad.size > 0:
            grad[:n] = (dc / state["scale"]).ravel()
            grad[n] = -1.0
        state["uo"][eta] = u_out
        if eta == ETA_I:  # intermediate tracks volume; rescale so it hits VOLFRAC
            state["vfrac_d"] *= VOLFRAC / x_r.mean()
            if args.animate:
                fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
                ax.imshow(x_r.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
                ax.axis("off")
                fig.tight_layout(pad=0)
                plt.savefig(ANIMATION_DIR / f"frame_{state['it']:d}.jpg")
                plt.close()
        return u_out / state["scale"] - z[n]

    return performance


def volume(z, grad):  # dilated volume <= rescaled bound (so intermediate hits VOLFRAC)
    beta = state["beta"]
    x_tilde = density_filter(z[:n].reshape(NX, NY))
    x_d = projection(x_tilde, beta, ETA_D)
    if grad.size > 0:
        grad[:n] = (filter_adjoint(dprojection(x_tilde, beta, ETA_D) / n) / state["vfrac_d"]).ravel()
        grad[n] = 0.0
    return x_d.mean() / state["vfrac_d"] - 1.0


performances = [make_performance(eta) for eta in ETAS]
lb = np.concatenate([np.zeros(n), [-1e2]])
ub = np.concatenate([np.ones(n), [1e2]])

tic = time.time()
pbar = tqdm(total=MAX_ITER)
zval = np.concatenate([np.full(n, VOLFRAC), [1.0]])
while state["it"] < MAX_ITER:
    opt = nlopt.opt(nlopt.LD_MMA, nvar)
    opt.set_min_objective(epigraph)
    for perf in performances:
        opt.add_inequality_constraint(perf, 0.0)
    opt.add_inequality_constraint(volume, 0.0)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    final = state["beta"] >= BETA_MAX and state["p"] >= PENAL
    opt.set_maxeval(MAX_ITER - state["it"] if final else CONT_STEP)
    if final:
        opt.set_xtol_abs(CHANGE_TOL)  # stop once the design settles
    zval = opt.optimize(zval)
    if final:
        break
    state["beta"] = min(2.0 * state["beta"], BETA_MAX)
    state["p"] = min(state["p"] + 0.5, PENAL)

xval = zval[:n]
beta = state["beta"]
it = state["it"]
toc = time.time()
print(
    f"elapsed time {toc - tic:.2f} s for {it} iter\n"
    f"time per iter {(toc - tic) / it:.2e} s"
)

# ---------------------------------- post-processing ----------------------------------
x_int = projection(density_filter(xval.reshape(NX, NY)), beta, ETA_I)
rho_thresh = (x_int > THRESHOLD).astype(float)

u = np.zeros(ndof)
factorize(EMIN + x_int**PENAL * (E0 - EMIN))
u[free] = solve(force_free)
u_out_phys = L @ u
factorize(EMIN + rho_thresh**PENAL * (E0 - EMIN))
u[free] = solve(force_free)
u_out_thresh = L @ u
print(
    f"intermediate u_out {u_out_phys:.3e} vol {x_int.mean():.3f}\n"
    f"thresholded  u_out {u_out_thresh:.3e} vol {rho_thresh.mean():.3f}"
)


def mirror(field):  # reflect the half-model across the bottom symmetry plane
    return np.concatenate([np.flip(field, axis=1), field], axis=1)


for field, name in ((x_int, "topopt_inverter"), (rho_thresh, "topopt_inverter_thresh")):
    full = mirror(field)
    fig, ax = plt.subplots(figsize=(NX / 100, 2 * NY / 100), dpi=150)
    ax.imshow(full.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0)
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

# x-displacement evaluated on the thresholded structure (void left transparent)
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
indicator = np.array(acc.data()[1])
tri = acc.triangulation()
tri.set_mask(indicator[tri.triangles].mean(axis=1) < 0.5)

fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
ax.tricontourf(tri, ux, cmap="turbo", levels=64)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0)
if args.book:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "topopt_inverter_ux.png", transparent=True)
    plt.close()
elif not args.animate:
    plt.show()
else:
    plt.close()
