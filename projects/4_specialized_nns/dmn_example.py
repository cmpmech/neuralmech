import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from NN import DMN, LaminateBlock, laminate_rotation
from postprocessing import save_csv
from solvers.homogenization import nonlinear_uniaxial_response
from solvers.material_subroutines.nonlinear_elastic import ABI, build

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)  # phase stiffnesses span orders of magnitude
torch.set_num_threads(1)  # tiny 3x3 matrices: threading is pure overhead
device = torch.device("cpu")  # faster on cpu, because matrices are small

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
DEPTH = 4
LR = 5e-3
EPOCHS = 4000

# define loss: scale-invariant fit error (a plain MSE lets the stiff samples dominate)
def cost_fun(pred, target):
    return (torch.linalg.matrix_norm(pred - target) / torch.linalg.matrix_norm(target)).mean()


# online nonlinear-elastic law sigma = C0 eps + c (eps.eps) eps applied to the frozen
# tree; phase 1 is a stiffening matrix, phase 2 a soft inclusion. The same parameters
# [E, nu, c] drive the finite-cell reference through the C subroutine.
MATRIX = [1.0, 0.3, 50.0]
INCLUSION = [0.2, 0.3, 0.0]
EPS_MAX = 0.08
NSTEPS = 50

# ------------------------------------- helper ----------------------------------------
def isotropic_stiffness(E, nu):
    return E / (1 - nu**2) * torch.tensor(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
    )


def phase_law(eps, C0, c):
    ee = eps @ eps
    sigma = C0 @ eps + c * ee * eps
    tangent = C0 + c * ee * torch.eye(3) + 2 * c * torch.outer(eps, eps)
    return sigma, tangent


# batched laminate homogenization over all samples at once (fast training path)
def block_homogenize(v1, alpha, C1, C2):
    v2 = 1 - v1
    A1, B1, D1 = C1[:, 0, 0], C1[:, 0, 1:], C1[:, 1:, 1:]
    A2, B2, D2 = C2[:, 0, 0], C2[:, 0, 1:], C2[:, 1:, 1:]
    D1i, D2i = torch.linalg.inv(D1), torch.linalg.inv(D2)
    Dti = torch.linalg.inv(v1 * D1i + v2 * D2i)
    P = v1 * torch.einsum("nij,nj->ni", D1i, B1) + v2 * torch.einsum("nij,nj->ni", D2i, B2)
    A_bar = (
        v1 * (A1 - torch.einsum("ni,nij,nj->n", B1, D1i, B1))
        + v2 * (A2 - torch.einsum("ni,nij,nj->n", B2, D2i, B2))
        + torch.einsum("ni,nij,nj->n", P, Dti, P)
    )
    B_bar = torch.einsum("nij,nj->ni", Dti, P)
    C_bar = torch.zeros_like(C1)
    C_bar[:, 0, 0], C_bar[:, 0, 1:], C_bar[:, 1:, 0], C_bar[:, 1:, 1:] = A_bar, B_bar, B_bar, Dti
    return (laminate_rotation(alpha) @ C_bar) @ laminate_rotation(-alpha)


def effective_stiffness(model, C1, C2):
    leaves = [C1 if n % 2 == 0 else C2 for n in range(2**model.depth)]
    for layer in model.layers:
        leaves = [
            block_homogenize(b.v1, b.alpha, leaves[2 * i], leaves[2 * i + 1])
            for i, b in enumerate(layer)
        ]
    return leaves[0]


# per-leaf-tangent tree pass for the online step: bottom-up stiffness, top-down strain
def dmn_tree(model, tangents, deps):
    caches, stiffness = [], tangents
    for layer in model.layers:
        homogenized, layer_cache = [], []
        for i, block in enumerate(layer):
            C, cache = block.homogenize_stiffness(stiffness[2 * i], stiffness[2 * i + 1])
            homogenized.append(C)
            layer_cache.append(cache)
        caches.append(layer_cache)
        stiffness = homogenized

    strains = [deps]
    for layer, layer_cache in zip(reversed(model.layers), reversed(caches)):
        strains = [
            child
            for block, cache, d in zip(layer, layer_cache, strains)
            for child in block.recover_strains(d, cache)
        ]
    return strains


# explicit incremental drive of a uniaxial macro-strain path through the frozen tree
def dmn_online(model):
    C0 = [isotropic_stiffness(*MATRIX[:2]), isotropic_stiffness(*INCLUSION[:2])]
    c = [MATRIX[2], INCLUSION[2]]
    leaf_eps = [torch.zeros(3) for _ in range(2**model.depth)]
    deps = torch.tensor([EPS_MAX / NSTEPS, 0.0, 0.0])

    eps_macro, sig_macro, curve = torch.zeros(3), torch.zeros(3), []
    for _ in range(NSTEPS):
        tangents = [phase_law(leaf_eps[n], C0[n % 2], c[n % 2])[1] for n in range(len(leaf_eps))]
        leaf_deps = dmn_tree(model, tangents, deps)

        leaf_dsig = []
        for n, dn in enumerate(leaf_deps):
            sig0 = phase_law(leaf_eps[n], C0[n % 2], c[n % 2])[0]
            leaf_eps[n] = leaf_eps[n] + dn
            sig1 = phase_law(leaf_eps[n], C0[n % 2], c[n % 2])[0]
            leaf_dsig.append(sig1 - sig0)

        eps_macro = eps_macro + deps
        sig_macro = sig_macro + model.homogenize_stress(leaf_dsig)
        curve.append((eps_macro[0].item(), sig_macro[0].item()))
    return np.array(curve)


# --------------------------------------- load data -----------------------------------
data = np.load(DATA_DIR / "hom_dmn.npz")
C1 = torch.tensor(data["C1"])
C2 = torch.tensor(data["C2"])
C_eff = torch.tensor(data["C_eff"])

# homogenization is degree-1 in the phase stiffnesses, so normalize each sample by |C1|
# to weight the fit uniformly across contrast levels
scale = torch.linalg.matrix_norm(C1)[:, None, None]
C1, C2, C_eff = C1 / scale, C2 / scale, C_eff / scale

ntrain = int(0.8 * len(C1))
C1_train, C2_train, C_eff_train = C1[:ntrain], C2[:ntrain], C_eff[:ntrain]
C1_val, C2_val, C_eff_val = C1[ntrain:], C2[ntrain:], C_eff[ntrain:]

# --------------------------- instantiate model & optimizer ---------------------------
model = DMN(DEPTH)
model.to(device)
# break the volume-fraction symmetry of the default all-equal init so the tree does
# not collapse to a single laminate chain
for block in model.modules():
    if isinstance(block, LaminateBlock):
        nn.init.normal_(block.v1_logit, std=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------- training -----------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    cost = cost_fun(effective_stiffness(model, C1_train, C2_train), C_eff_train)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    model.eval()
    with torch.no_grad():
        val_cost[epoch] = cost_fun(effective_stiffness(model, C1_val, C2_val), C_eff_val).item()

    if epoch % 50 == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

torch.save(model, BASE_DIR / f"../../models/dmn_{DEPTH}.pt2")

# ----------------------------------- postprocessing ----------------------------------
model.eval()
with torch.no_grad():
    C_dmn = effective_stiffness(model, C1_val, C2_val)
rel_err = (torch.linalg.matrix_norm(C_dmn - C_eff_val) / torch.linalg.matrix_norm(C_eff_val)).mean()
print(f"validation effective-stiffness relative error {rel_err:.2e}")

with torch.no_grad():
    dmn_curve = dmn_online(model)
fe_curve = nonlinear_uniaxial_response(
    build().material_address, MATRIX, INCLUSION, ABI, eps_max=EPS_MAX, nsteps=NSTEPS
)

triu = np.triu_indices(3)
fe_entries = C_eff_val[:, triu[0], triu[1]].flatten().numpy()
dmn_entries = C_dmn[:, triu[0], triu[1]].flatten().numpy()

if not args.book:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot([fe_entries.min(), fe_entries.max()], [fe_entries.min(), fe_entries.max()], "k")
    ax1.plot(fe_entries, dmn_entries, "r.")
    ax1.set_xlabel("FE effective stiffness")
    ax1.set_ylabel("DMN effective stiffness")
    ax2.plot(fe_curve[:, 0], fe_curve[:, 1], "k")
    ax2.plot(dmn_curve[:, 0], dmn_curve[:, 1], "r")
    ax2.set_xlabel("macro strain e11")
    ax2.set_ylabel("macro stress s11")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "dmn_parity.csv", fe=fe_entries, dmn=dmn_entries)
    save_csv(CSV_DIR / "dmn_nonlinear.csv", eps=fe_curve[:, 0], fe=fe_curve[:, 1], dmn=dmn_curve[:, 1])
