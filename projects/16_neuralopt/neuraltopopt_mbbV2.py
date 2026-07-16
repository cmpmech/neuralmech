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
import numpy as np
import scipy.ndimage
import scipy.sparse
import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames/neuraltopopt_mbb"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

device = torch.device("cpu")
SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ------------------------------------- resolution ------------------------------------
N = 96
ANSATZ = "dcn"  # dcn or linear

# -------------------------------------- settings -------------------------------------
# geometry
LENGTHS = [3.0, 1.0]

# discretization
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

# post-processing
THRESHOLD = 0.5

# optimization (per-ansatz lr and polynomial lr decay (BETA * iter + 1) ** ALPHA)
MAX_ITER = 150
CLIP = 0.1  # gradient-norm clipping
HYPERPARAMS = {
    "dcn": dict(lr=5e-3, alpha=-0.5, beta=0.2),
    "linear": dict(lr=5e-2, alpha=0.0, beta=0.0),
}
LR = HYPERPARAMS[ANSATZ]["lr"]
ALPHA = HYPERPARAMS[ANSATZ]["alpha"]
BETA = HYPERPARAMS[ANSATZ]["beta"]


# ----------------------------------- design ansatz -----------------------------------
# the classes below are copied verbatim from reference/NeuralNetwork.py so the design
# parameterization matches the GDTopOptDriver exactly; later these are to be replaced
# with the shared helpers in NN.py
class gaussian(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x**2) / (2 * self.sigma**2))


class noActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class pixelNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(
            torch.sum(x**2, axis=(2, 3), keepdim=True) / x.shape[2] / x.shape[3] + 1e-8
        )


def makeCNNBlocks(
    channels,
    numberOfConvolutionsPerBlock,
    kernelSize,
    activation,
    normalization,
    skipChannels=None,
    lastActivation=True,
):
    padding = (kernelSize - 1) // 2
    if skipChannels == None:
        skipChannels = [0 for i in channels]

    convolutions = torch.nn.ModuleList()
    normalizations = torch.nn.ModuleList()
    activations = torch.nn.ModuleList()
    for i in range(len(channels) - 1):
        convolutions.append(
            torch.nn.Conv2d(
                channels[i] + skipChannels[i],
                channels[i + 1],
                kernelSize,
                stride=1,
                padding=padding,
            )
        )
        normalizations.append(normalization(channels[i] + skipChannels[i]))
        if i < len(channels) - 2 or numberOfConvolutionsPerBlock != 1:
            activations.append(activation())
        else:
            activations.append(noActivation())

        for j in range(numberOfConvolutionsPerBlock - 1):
            convolutions.append(
                torch.nn.Conv2d(
                    channels[i + 1],
                    channels[i + 1],
                    kernelSize,
                    stride=1,
                    padding=padding,
                )
            )
            normalizations.append(normalization(channels[i + 1]))
            if i < len(channels) - 2 and j < numberOfConvolutionsPerBlock - 2:
                activations.append(activation())
            else:
                activations.append(noActivation())

    for i in range(len(convolutions)):
        torch.nn.init.xavier_normal_(convolutions[i].weight)
        convolutions[i].bias.data.fill_(0)
    if lastActivation == False:
        torch.nn.init.normal_(convolutions[-1].weight, std=0.01)

    return convolutions, normalizations, activations


class generator(torch.nn.Module):
    def __init__(
        self,
        channels,
        channelsOut,
        numberOfConvolutionsPerBlock,
        kernelSize,
        volumeFraction,
        Nx,
        Ny,
        numberOfVoxels,
    ):
        super().__init__()
        # hyperparameters (more choices below regarding output/activation/normalization)
        self.volumeFraction = volumeFraction  # potentially unused
        self.kernelSize = kernelSize
        self.channels = channels  # channelsIn is defined implicitly
        self.channelsOut = channelsOut
        self.numberOfConvolutionsPerBlock = numberOfConvolutionsPerBlock
        self.sigmaForGaussian = 0.5

        self.activation = lambda: gaussian(
            self.sigmaForGaussian
        )  # torch.nn.SiLU(), torch.nn.LeakyReLU() and torch.nn.PReLU() seem worse
        # self.activation = lambda : makeAdaptiveActivation(10, gaussian(self.sigmaForGaussian))
        self.finalActivation = torch.nn.Softmax(dim=1)
        self.normalization = lambda s: torch.nn.BatchNorm2d(s)  # pixelNorm() is worse

        # upsampling
        self.convolutions, self.normalizations, self.activations = makeCNNBlocks(
            self.channels,
            self.numberOfConvolutionsPerBlock,
            self.kernelSize,
            self.activation,
            self.normalization,
            lastActivation=False,
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )  # bilinear is worse

        self.input = torch.randn(
            (
                1,
                self.channels[0],
                int(Nx * numberOfVoxels / 2 ** (len(self.channels) - 1)),
                int(Ny * numberOfVoxels / 2 ** (len(self.channels) - 1)),
            )
        )
        self.input = self.input / (torch.max(self.input) - torch.min(self.input)) * 2

    def forward(self, x):
        x = self.input
        for i in range(len(self.channels) - 1):
            x = self.upsample(x)
            for j in range(self.numberOfConvolutionsPerBlock):
                index = i * self.numberOfConvolutionsPerBlock + j
                x = self.activations[index](
                    self.convolutions[index](self.normalizations[index](x))
                )

        x = self.finalActivation(x)
        return x[:, 0:1, :, :]


class ConstantAnsatzFEM(torch.nn.Module):
    def __init__(self, Nx, Ny, numberOfVoxels, volumeFraction):
        super().__init__()

        self.coeff = torch.nn.Parameter(
            torch.ones((1, 1, Nx * numberOfVoxels, Ny * numberOfVoxels))
            * volumeFraction
        )
        self.volumeFraction = volumeFraction

    def forward(self, dummy):
        return self.coeff


# every ansatz exposes parameters and a forward() returning a (1, 1, NX, NY) density
if ANSATZ == "dcn":  # convolutional generator: fixed latent image -> density field
    assert int(NX) % 8 == 0 and int(NY) % 8 == 0, (
        f"generator upsamples 3x; design grid {[int(NX), int(NY)]} must be divisible by 8"
    )
    KERNEL_SIZE = 5
    CHANNELS = [16, 8, 4, 2]
    CHANNELS_OUT = 2
    N_CONV_PER_BLOCK = 1
    model = generator(
        CHANNELS,
        CHANNELS_OUT,
        N_CONV_PER_BLOCK,
        KERNEL_SIZE,
        VOLFRAC,
        int(NX),
        int(NY),
        1,
    ).to(device)
    params = list(model.parameters())
    forward = lambda: model(None)

elif ANSATZ == "linear":  # no network: voxel densities are the design variables
    model = ConstantAnsatzFEM(int(NX), int(NY), 1, VOLFRAC).to(device)
    params = list(model.parameters())
    forward = lambda: model(None)

optimizer = torch.optim.Adam(params, lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda it: (BETA * it + 1) ** ALPHA
)

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
    basis_local, integrand, quadrature, mlhp.absoluteQuadratureOrder([QUAD_ORDER, QUAD_ORDER])
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

    def assemble_K_free(rho_field, penal):  # penalised stiffness on the free dofs
        E_e = EMIN + grid_to_elements(rho_field) ** penal * (E0 - EMIN)
        K_e = np.einsum("es,sij->eij", E_e, K_locals, optimize=True)
        data = np.add.reduceat(K_e.ravel()[data_idx], seg)
        return scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=(free.size, free.size)
        )

    return assemble_K_free


assemble_K_free = build_assemble_K_free()


# for CHOLMOD: the SPD system's sparsity is factored symbolically once
K_free = assemble_K_free(np.full((NX, NY), VOLFRAC), PENAL0)
A = cvxopt.spmatrix(
    cvxopt.matrix(K_free.data),
    cvxopt.matrix(K_free.indices.tolist()),
    cvxopt.matrix(np.repeat(np.arange(free.size), np.diff(K_free.indptr)).tolist()),
    (free.size, free.size),
)
factor = cvxopt.cholmod.symbolic(A)


def solve_free(rho_field, penal):
    A.V = cvxopt.matrix(assemble_K_free(rho_field, penal).data)
    cvxopt.cholmod.numeric(A, factor)
    b = cvxopt.matrix(force_free)
    cvxopt.cholmod.solve(factor, b)
    return np.array(b).ravel()


# ----------------------------------- density filter ----------------------------------
ceil_r = int(np.ceil(RMIN))
ky, kx = np.meshgrid(np.arange(-ceil_r, ceil_r + 1), np.arange(-ceil_r, ceil_r + 1))
kernel = np.maximum(0.0, RMIN - np.sqrt(kx**2 + ky**2))
Hs = scipy.ndimage.convolve(np.ones((NX, NY)), kernel, mode="constant", cval=0.0)


def filter_sensitivity(rho, dc):
    num = scipy.ndimage.convolve(rho * dc, kernel, mode="constant", cval=0.0)
    return num / (np.maximum(rho, 1e-3) * Hs)


# ------------------------------------ optimization -----------------------------------
penal = PENAL0
penalty = PENALTY0
compliance0 = None

tic = time.time()
pbar = tqdm(range(MAX_ITER))
for it in pbar:
    rho_ = forward()
    rho = rho_[0, 0].detach().numpy()

    u = np.zeros(ndof)
    u[free] = solve_free(rho, penal)
    compliance = force @ u
    if compliance0 is None:
        compliance0 = compliance  # normalise the compliance sensitivity once

    # compliance sensitivity, mapped back to the design grid and filtered
    ue = u[efts]
    ce = np.einsum("ei,sij,ej->es", ue, K_locals, ue, optimize=True)
    dc = -penal * rho ** (penal - 1) * (E0 - EMIN) * elements_to_grid(ce)
    dc = filter_sensitivity(rho, dc)

    # quadratic volume penalty: (mean_rho / VOLFRAC - 1) ** 2, weight grows each iter
    mean_rho = rho.mean()
    dv = 2 * (mean_rho / VOLFRAC - 1) / (VOLFRAC * NX * NY)
    sensitivity = dc / compliance0 + penalty * dv

    optimizer.zero_grad()
    rho_.backward(torch.from_numpy(sensitivity).unsqueeze(0).unsqueeze(0))
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    optimizer.step()
    scheduler.step()
    if ANSATZ == "linear":
        model.coeff.data.clamp_(0.0, 1.0)  # keep design variables in [0, 1]

    penal = min(penal + PENAL_INC, PENAL_MAX)
    penalty = min(penalty + PENALTY_INC, PENALTY_MAX)

    pbar.set_postfix(c=f"{compliance:.2e}", vol=f"{mean_rho:.3f}", p=f"{penal:.2f}")

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s for {MAX_ITER} iter")

# ---------------------------------- post-processing ----------------------------------
rho_thresh = (rho > THRESHOLD).astype(float)

u = np.zeros(ndof)
u[free] = solve_free(rho_thresh, penal)
compliance_thresh = force @ u
print(f"thresholded  c {compliance_thresh:.3e} vol {rho_thresh.mean():.3f}")

for field, name in ((rho, "topopt_mbb"), (rho_thresh, "topopt_mbb_thresh")):
    fig, ax = plt.subplots(figsize=(NX / 100, NY / 100), dpi=150)
    ax.imshow(field.T, origin="lower", cmap="binary", vmin=0.0, vmax=1.0, alpha=field.T)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    if args.book:
        plt.savefig(RGB_PDF_DIR / f"{name}.pdf", transparent=True)
        plt.close()
    elif not args.animate:
        plt.show()
    else:
        plt.close()
