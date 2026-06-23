import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from NN import BayesianMLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 5000
LR = 1e-2
PRIOR_STD = 1.0  # acts as (inverse) L2 regularization
MC_SAMPLES = 10
KL_WEIGHT = 0.5
NOISE_STD = 0.07
SAMPLES = 32
INF_SAMPLES = 100


# define loss
def elbo(model, x, y, prior_std, noise_std, mc_samples, kl_weight):
    nll = 0
    for _ in range(mc_samples):
        y_pred = model(x)
        nll += F.mse_loss(y_pred, y, reduction="mean") / (2 * noise_std**2)
    kl = model.kl_divergence(prior_std)
    loss = nll / mc_samples + kl_weight * kl / len(x)
    return loss, nll / mc_samples, kl / len(x)


# model settings
LAYERS = [1, 32, 32, 32, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------ create data ------------------------------------
x_train1 = torch.rand(SAMPLES // 2) * (-3)
x_train2 = torch.rand(SAMPLES // 2) * 1.5 + 1.5
x_train = torch.cat([x_train1, x_train2], dim=0).reshape(-1, 1)
y_train = torch.sin(x_train) + torch.randn_like(x_train) * NOISE_STD
x_train, y_train = x_train.to(device), y_train.to(device)

# --------------------------- instantiate model & optimizer ---------------------------
model = BayesianMLP(LAYERS, post_modules=ACTIVATIONS).to(device)
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
print_every = 10

tic = time.time()
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    cost, nll, kl = elbo(
        model, x_train, y_train, PRIOR_STD, NOISE_STD, MC_SAMPLES, KL_WEIGHT
    )
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.detach().item()

    if epoch % print_every == 0:
        pbar.set_postfix(
            {
                "loss": f"{train_cost[epoch]:.2e}",
                "nll": f"{nll.detach():.2e}",
                "kl": f"{kl.detach():.2e}",
            }
        )

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-6, 6, 200).reshape(-1, 1).to(device)
    y_preds = torch.stack([model(x_test) for _ in range(INF_SAMPLES)])
    mean = y_preds.mean(dim=0)
    std = y_preds.std(dim=0)  # epistemic
total_std = torch.sqrt(std**2 + NOISE_STD**2)

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(train_cost, "k")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_train.cpu(), y_train.cpu(), "bo")
    ax.plot(x_test.squeeze().cpu(), mean.squeeze().cpu(), "k")
    ax.fill_between(
        x_test.squeeze().cpu(),
        (mean - 2 * total_std).squeeze().cpu(),
        (mean - 2 * std).squeeze().cpu(),
        alpha=0.3,
        color="b",
    )
    ax.fill_between(
        x_test.squeeze().cpu(),
        (mean + 2 * std).squeeze().cpu(),
        (mean + 2 * total_std).squeeze().cpu(),
        alpha=0.3,
        color="b",
    )
    ax.fill_between(
        x_test.squeeze().cpu(),
        (mean - 2 * std).squeeze().cpu(),
        (mean + 2 * std).squeeze().cpu(),
        alpha=0.3,
        color="r",
    )
    ax.set_ylim(-2, 2)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "vainf.csv",
        x=x_test.squeeze().cpu().numpy(),
        mean=mean.squeeze().cpu().numpy(),
        std=std.squeeze().cpu().numpy(),
        std_total=total_std.squeeze().cpu().numpy(),
    )
    save_csv(
        RESULTS_DIR / "vainf_train.csv",
        x=x_train.squeeze().cpu().numpy(),
        y=y_train.squeeze().cpu().numpy(),
    )
