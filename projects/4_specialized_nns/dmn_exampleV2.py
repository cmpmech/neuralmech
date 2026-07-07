import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from NN import DMN

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True  # TODO check
torch.set_default_dtype(torch.float64)  # phase stiffnesses span orders of magnitude
torch.set_num_threads(1)  # tiny 3x3 matrices: threading is pure overhead
device = torch.device("cpu")  # small matrices, faster on cpu

# -------------------------------------- settings -------------------------------------
# hyperparameters
DEPTH = 4
LR = 1e-2  # 5e-3  # 1e-2 also works?
EPOCHS = 2000  # 2000  # LEAVE THIS FOR NOW


# define loss: scale-invariant relative fit error (a plain MSE lets stiff samples dominate)
def cost_fun(pred, target):
    return (
        torch.linalg.matrix_norm(pred - target) / torch.linalg.matrix_norm(target)
    ).mean()


# ------------------------------------- helper ----------------------------------------
def isotropic_stiffness(E, nu):
    return (
        E / (1 - nu**2) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    )


def phase_law(eps, C0, c):
    # nonlinear-elastic response and tangent per leaf: eps (K, 3), C0 (K, 3, 3), c (K,)
    ee = (eps * eps).sum(-1)
    sigma = torch.einsum("kij,kj->ki", C0, eps) + c[:, None] * ee[:, None] * eps
    tangent = (
        C0
        + (c * ee)[:, None, None] * torch.eye(3)
        + 2 * c[:, None, None] * torch.einsum("ki,kj->kij", eps, eps)
    )
    return sigma, tangent


# --------------------------------------- load data -----------------------------------
data = np.load(DATA_DIR / "hom_dmn.npz")
C1 = torch.tensor(data["C1"]).to(device)
C2 = torch.tensor(data["C2"]).to(device)
C_eff = torch.tensor(data["C_eff"]).to(device)

# nonlinear-elastic law + FE reference curve generated alongside the dataset; the online
# test below applies the same per-phase law (E, nu, c) to the trained tree
MATRIX = tuple(data["matrix"])
INCLUSION = tuple(data["inclusion"])
EPS_MAX = float(data["eps_max"])
NSTEPS = int(data["nsteps"])
ref_eps, ref_sig = data["ref_eps"], data["ref_sig"]

ntrain = int(0.8 * len(C1))
C1_train, C2_train, C_eff_train = C1[:ntrain], C2[:ntrain], C_eff[:ntrain]
C1_val, C2_val, C_eff_val = C1[ntrain:], C2[ntrain:], C_eff[ntrain:]

# --------------------------- instantiate model & optimizer ---------------------------
model = DMN(DEPTH)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ---------------------------------------- training -----------------------------------
# the DMN forward is batched over samples; only C_root is needed for the stiffness fit
# (deps is an unused placeholder)
def predict(C1, C2):
    deps = torch.zeros(len(C1), 3)
    return model(C1, C2, deps)[0]


train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    cost = cost_fun(predict(C1_train, C2_train), C_eff_train)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    model.eval()
    with torch.no_grad():
        val_cost[epoch] = cost_fun(predict(C1_val, C2_val), C_eff_val).item()

    if epoch % 50 == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------------ nonlinear test ---------------------------------
# drive a uniaxial macro-strain path through the trained tree with the nonlinear law per
# phase (no retraining); leaves alternate matrix / inclusion like the linear forward, so
# each leaf carries its own strain-dependent tangent
K = 2**DEPTH
C0 = torch.stack(
    [
        isotropic_stiffness(*MATRIX[:2])
        if k % 2 == 0
        else isotropic_stiffness(*INCLUSION[:2])
        for k in range(K)
    ]
)
c = torch.tensor([MATRIX[2] if k % 2 == 0 else INCLUSION[2] for k in range(K)])

leaf_eps = torch.zeros(K, 3)
deps = torch.tensor([[EPS_MAX / NSTEPS, 0.0, 0.0]])
eps_macro = torch.zeros(1, 3)
sig_macro = torch.zeros(1, 3)
curve = [(0.0, 0.0)]

model.eval()
with torch.no_grad():
    for _ in range(NSTEPS):
        # frozen-tangent increment: distribute deps to the leaves, update, homogenize dsig
        sig0, tangent = phase_law(leaf_eps, C0, c)
        leaf_deps = model.homogenize([tangent[k : k + 1] for k in range(K)], deps)[1]
        leaf_eps = leaf_eps + torch.cat(leaf_deps)
        sig1 = phase_law(leaf_eps, C0, c)[0]
        sig_macro = sig_macro + model.homogenize_stress(
            [(sig1 - sig0)[k : k + 1] for k in range(K)]
        )
        eps_macro = eps_macro + deps
        curve.append((eps_macro[0, 0].item(), sig_macro[0, 0].item()))
curve = np.array(curve)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(ref_eps, ref_sig, "k")  # nonlinear FE reference
ax.plot(curve[:, 0], curve[:, 1], "r")  # DMN online prediction
ax.set_xlabel("macro strain e11")
ax.set_ylabel("macro stress s11")
plt.show()
