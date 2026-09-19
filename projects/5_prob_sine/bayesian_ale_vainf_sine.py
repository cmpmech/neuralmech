import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP, BayesianLinear, BayesianMLP

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 3000  # 2000  # 5000
VAR_EPOCHS = 2000
BAY_EPOCHS = 5000
VAR_REGULARIZATION = 1e-2  # keeps the variance net from interpolating single residuals
LR = 2e-2
PRIOR_STD = 1.0  # acts as (inverse) L2 regularization
MC_SAMPLES = 40
KL_WEIGHT = 0.5
NOISE_STD = 0.1
SAMPLES = 32
INF_SAMPLES = 100

print_every = 10


# define loss
def train(model, loss_fun, epochs, regularization=0):
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=regularization)
    cost = [0] * epochs
    pbar = tqdm(range(epochs))
    model.train()
    for epoch in pbar:
        optimizer.zero_grad()
        loss = loss_fun()
        loss.backward()
        optimizer.step()
        cost[epoch] = loss.item()

        if epoch % print_every == 0:
            pbar.set_postfix({"loss": f"{cost[epoch]:.2e}"})
    return cost


def gamma_nll(y_pred, r):
    alpha, lam = F.softplus(y_pred[:, 0:1]), F.softplus(y_pred[:, 1:2])
    nll = (
        torch.lgamma(alpha)
        - alpha * torch.log(lam)
        - (alpha - 1) * torch.log(r)
        + lam * r
    )
    return torch.mean(nll)


# softplus rather than exp positivity, so that the variance grows at most linearly
# outside the data support instead of exploding
def aleatoric_variance(y_pred):
    return F.softplus(y_pred[:, 0:1]) / F.softplus(y_pred[:, 1:2])


def elbo(model, x, y, var, prior_std, mc_samples, kl_weight):
    nll = 0
    for _ in range(mc_samples):
        y_pred = model(x)
        nlls = 0.5 * torch.log(2 * torch.pi * var) + (y - y_pred) ** 2 / (2 * var)
        nll += torch.mean(nlls)
    kl = model.kl_divergence(prior_std)
    return nll / mc_samples + kl_weight * kl / len(x)


# model settings
LAYERS = [1, 32, 32, 32, 1]
VAR_LAYERS = [1, 16, 2]  # gamma shape and rate
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]
VAR_ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(VAR_LAYERS) - 2)]

# ------------------------------------ create data ------------------------------------
target = lambda x: torch.sin(x)
noise_std = lambda x: (x / 6 + 1) * NOISE_STD
noise = lambda x: torch.randn_like(x) * noise_std(x)  # mean 0

x_train1 = torch.rand(SAMPLES // 2) * (-3)
x_train2 = torch.rand(SAMPLES // 2) * 1.5 + 1.5
x_train = torch.cat([x_train1, x_train2], dim=0).reshape(-1, 1)
y_train = target(x_train) + noise(x_train)
x_train, y_train = x_train.to(device), y_train.to(device)

# -------------------------------- mean network training -------------------------------
mean_model = MLP(LAYERS, post_modules=ACTIVATIONS).to(device)
init_weights(mean_model, ACTIVATIONS[0])

tic = time.time()
mean_cost = train(mean_model, lambda: F.mse_loss(mean_model(x_train), y_train), EPOCHS)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------ variance network training -----------------------------
mean_model.eval()
with torch.no_grad():
    residual = (mean_model(x_train) - y_train) ** 2
residual = residual.clamp_min(1e-12)  # log r diverges where the mean fits exactly

var_model = MLP(VAR_LAYERS, post_modules=VAR_ACTIVATIONS).to(device)
init_weights(var_model, VAR_ACTIVATIONS[0])

tic = time.time()
var_cost = train(
    var_model,
    lambda: gamma_nll(var_model(x_train), residual),
    VAR_EPOCHS,
    VAR_REGULARIZATION,
)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------- bayesian network training ------------------------------
var_model.eval()
with torch.no_grad():
    var_train = aleatoric_variance(var_model(x_train))

model = BayesianMLP(LAYERS, post_modules=ACTIVATIONS).to(device)

# warm start the variational means with the deterministic mean network
linears = [m for m in mean_model.modules() if isinstance(m, nn.Linear)]
bayesians = [m for m in model.modules() if isinstance(m, BayesianLinear)]
for linear, bayesian in zip(linears, bayesians):
    bayesian.weight_mu.data.copy_(linear.weight)
    bayesian.bias_mu.data.copy_(linear.bias)

tic = time.time()
bayesian_cost = train(
    model,
    lambda: elbo(model, x_train, y_train, var_train, PRIOR_STD, MC_SAMPLES, KL_WEIGHT),
    BAY_EPOCHS,
)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-6, 6, 200).reshape(-1, 1).to(device)

model.eval()
with torch.no_grad():
    y_preds = torch.stack([model(x_test) for _ in range(INF_SAMPLES)])
    mean = y_preds.mean(dim=0)
    epistemic_std = y_preds.std(dim=0)
    aleatoric_std = torch.sqrt(aleatoric_variance(var_model(x_test)))
total_std = torch.sqrt(aleatoric_std**2 + epistemic_std**2)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(mean_cost, "k")
ax.plot(var_cost, "r")
ax.plot(bayesian_cost, "b")
plt.show()

fig, axes = plt.subplots(1, 2)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ground truth
axes[0].plot(x_test.squeeze().cpu(), target(x_test).squeeze().cpu(), "k")
axes[0].fill_between(
    x_test.squeeze().cpu(),
    (target(x_test) - 2 * noise_std(x_test)).squeeze().cpu(),
    (target(x_test) + 2 * noise_std(x_test)).squeeze().cpu(),
    alpha=0.3,
    color="r",
)

# prediction with separated aleatoric and epistemic uncertainty
axes[1].plot(x_test.squeeze().cpu(), mean.squeeze().cpu(), "k")
axes[1].fill_between(
    x_test.squeeze().cpu(),
    (mean - 2 * total_std).squeeze().cpu(),
    (mean - 2 * aleatoric_std).squeeze().cpu(),
    alpha=0.3,
    color="b",
)
axes[1].fill_between(
    x_test.squeeze().cpu(),
    (mean + 2 * aleatoric_std).squeeze().cpu(),
    (mean + 2 * total_std).squeeze().cpu(),
    alpha=0.3,
    color="b",
)
axes[1].fill_between(
    x_test.squeeze().cpu(),
    (mean - 2 * aleatoric_std).squeeze().cpu(),
    (mean + 2 * aleatoric_std).squeeze().cpu(),
    alpha=0.3,
    color="r",
)

for ax in axes:
    ax.plot(x_train.cpu(), y_train.cpu(), "ko")
    ax.set_ylim(-2, 2)
plt.show()
