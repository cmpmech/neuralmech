import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")  # TODO could be animated
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 4000
LR = 5e-3

# microstructure: two isotropic phases, plane strain, fixed inclusion fraction
NU = 0.3
E_MATRIX = 1.0
CI = 0.3  # inclusion volume fraction targeted by Mori-Tanaka
DEPTH = 3  # binary tree depth -> 2**DEPTH leaves alternating phase

# training contrasts: stiff inclusion sampled over a range of stiffness ratios
SAMPLES = 24
E_INCL_MIN, E_INCL_MAX = 3.0, 100.0

# online nonlinear prediction (no retraining)
E_INCL_ONLINE = 50.0
ALPHA = 40.0  # matrix stiffening strength in the hyperelastic potential
EPS_MAX = 0.05
STEPS = 40
NEWTON_ITERS = 30
NEWTON_TOL = 1e-10

# select rows {xx, xy} of a Voigt vector (the laminate continuity components)
P = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# place the jump amplitudes (a_xx, a_xy) back into a full Voigt strain jump
Q = P.t()

# --------------------------------------- helper --------------------------------------
def iso_stiffness(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return torch.tensor(
        [[lam + 2 * mu, lam, 0.0], [lam, lam + 2 * mu, 0.0], [0.0, 0.0, mu]]
    )


def voigt_rotation(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c * c, s * s, 2 * s * c]),
            torch.stack([s * s, c * c, -2 * s * c]),
            torch.stack([-s * c, s * c, c * c - s * s]),
        ]
    )


def homogenize(C1, C2, f1):
    # rank-1 laminate with normal e1: traction (xx, xy) continuous, eps_yy continuous
    f2 = 1 - f1
    G = f2 * C1 + f1 * C2
    J = -Q @ torch.linalg.solve(P @ G @ Q, P @ (C1 - C2))
    return (f1 * C1 + f2 * C2) + f1 * f2 * (C1 - C2) @ J


def mori_tanaka(Ci, Cm, ci, nu):
    d = 8 * (1 - nu)
    S = np.array(
        [
            [(5 - 4 * nu) / d, (4 * nu - 1) / d, 0.0],
            [(4 * nu - 1) / d, (5 - 4 * nu) / d, 0.0],
            [0.0, 0.0, 2 * (3 - 4 * nu) / d],
        ]
    )
    I = np.eye(3)
    A = np.linalg.inv(I + S @ np.linalg.inv(Cm) @ (Ci - Cm))
    A_mt = A @ np.linalg.inv((1 - ci) * I + ci * A)
    return Cm + ci * (Ci - Cm) @ A_mt


def stiff_law(e, C):
    return C @ e, C


def soft_law(e, K, mu, alpha):
    # small-strain hyperelastic potential W = 0.5 K tr^2 + mu p (1 + alpha p)
    tr = e[0] + e[1]
    grad_p = torch.stack([e[0] - e[1], e[1] - e[0], e[2]])  # dp/de, p = deviatoric norm
    p = 0.5 * (grad_p[0] * (e[0] - e[1]) + e[2] * e[2])
    vol = torch.tensor([1.0, 1.0, 0.0])
    stress = K * tr * vol + (mu + 2 * mu * alpha * p) * grad_p
    H_dev = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tangent = (
        K * torch.outer(vol, vol)
        + (mu + 2 * mu * alpha * p) * H_dev
        + 2 * mu * alpha * torch.outer(grad_p, grad_p)
    )
    return stress, tangent


def tree_stiffness(node):
    # linear homogenization: returns the macroscopic stiffness of the subtree
    if node >= 2**DEPTH:
        return C_PHASE[(node - 2**DEPTH) % 2]
    R = voigt_rotation(ANGLES[node - 1])
    f1 = torch.sigmoid(FRAC_LOGITS[node - 1])
    C = homogenize(tree_stiffness(2 * node), tree_stiffness(2 * node + 1), f1)
    return R @ C @ R.t()


def tree_response(node, e):
    # nonlinear propagation: returns (stress, tangent) given the subtree's strain
    if node >= 2**DEPTH:
        return LEAF_LAW[(node - 2**DEPTH) % 2](e)
    R = voigt_rotation(ANGLES[node - 1])
    f1 = torch.sigmoid(FRAC_LOGITS[node - 1])
    e_local = R.t() @ e
    a = torch.zeros(2)
    for _ in range(NEWTON_ITERS):
        s1, T1 = tree_response(2 * node, e_local + (1 - f1) * (Q @ a))
        s2, T2 = tree_response(2 * node + 1, e_local - f1 * (Q @ a))
        residual = P @ (s1 - s2)
        if residual.norm() < NEWTON_TOL:
            break
        a = a - torch.linalg.solve(P @ ((1 - f1) * T1 + f1 * T2) @ Q, residual)
    dj = -Q @ torch.linalg.solve(P @ ((1 - f1) * T1 + f1 * T2) @ Q, P @ (T1 - T2))
    s_local = f1 * s1 + (1 - f1) * s2
    T_local = (f1 * T1 + (1 - f1) * T2) + f1 * (1 - f1) * (T1 - T2) @ dj
    return R @ s_local, R @ T_local @ R.t()


# ------------------------------------ create data ------------------------------------
Cm = iso_stiffness(E_MATRIX, NU)
E_incl = torch.logspace(np.log10(E_INCL_MIN), np.log10(E_INCL_MAX), SAMPLES)
C_incl = [iso_stiffness(E, NU) for E in E_incl]
C_target = [
    torch.from_numpy(mori_tanaka(Ci.numpy(), Cm.numpy(), CI, NU)).float() for Ci in C_incl
]

# --------------------------- instantiate model & optimizer ---------------------------
N_INTERNAL = 2**DEPTH - 1
ANGLES = torch.nn.Parameter(torch.rand(N_INTERNAL) * np.pi)
FRAC_LOGITS = torch.nn.Parameter(torch.zeros(N_INTERNAL))
optimizer = torch.optim.AdamW([ANGLES, FRAC_LOGITS], lr=LR)

# -------------------------------------- training -------------------------------------
train_cost = [0.0] * EPOCHS
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    optimizer.zero_grad()
    cost = torch.zeros(())
    for Ci, Ct in zip(C_incl, C_target):
        C_PHASE = [Ci, Cm]
        cost = cost + ((tree_stiffness(1) - Ct) ** 2).sum() / (Ct**2).sum()
    cost = cost / SAMPLES
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()
    if epoch % 100 == 0:
        pbar.set_postfix({"train": f"{cost.item():.2e}"})

# ----------------------------------- postprocessing ----------------------------------
# online: stiff inclusion stays linear, matrix becomes hyperelastic, no retraining
C_stiff = iso_stiffness(E_INCL_ONLINE, NU)
lam = E_MATRIX * NU / ((1 + NU) * (1 - 2 * NU))
mu = E_MATRIX / (2 * (1 + NU))
LEAF_LAW = [
    lambda e: stiff_law(e, C_stiff),
    lambda e: soft_law(e, lam + mu, mu, ALPHA),
]

C_PHASE = [C_stiff, Cm]
C_linear = tree_stiffness(1).detach()

eps = torch.linspace(0, EPS_MAX, STEPS)
sigma_nonlinear = torch.zeros(STEPS)
sigma_linear = torch.zeros(STEPS)
with torch.no_grad():
    for i, e in enumerate(eps):
        ebar = torch.stack([e, torch.zeros(()), torch.zeros(())])
        sigma_nonlinear[i] = tree_response(1, ebar)[0][0]
        sigma_linear[i] = (C_linear @ ebar)[0]

if not args.book:
    fig, ax = plt.subplots(1, 2)
    ax[0].set_yscale("log")
    ax[0].plot(train_cost, "k")
    ax[1].plot(eps, sigma_nonlinear, "k")
    ax[1].plot(eps, sigma_linear, "b")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "dmn_loss.csv",
        epoch=np.arange(EPOCHS),
        cost=np.array(train_cost),
    )
    save_csv(
        RESULTS_DIR / "dmn_stress_strain.csv",
        eps=eps.numpy(),
        nonlinear=sigma_nonlinear.numpy(),
        linear=sigma_linear.numpy(),
    )
