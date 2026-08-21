import argparse
import time
import warnings
from pathlib import Path

import hamiltorch
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from DL import init_weights
from NN import MLP
from postprocessing import save_csv

warnings.filterwarnings(
    "ignore", message=".*Converting a tensor with requires_grad=True to a scalar.*"
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 2000  # mean
VAR_EPOCHS = 2000  # aleatoric std
VAR_REGULARIZATION = 1e-2
LR = 2e-2
HMC_SAMPLES = 500  # epistemic std
BURN_IN = 50  # samples to discard (remove bias from initialization)
HMC_STEP_SIZE = 0.002  # lower: more stable, higher: faster exploration
HMC_STEPS_PER_SAMPLE = 15
PRIOR_STD = 1.0
NOISE_STD = 0.1
SAMPLES = 32

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
def aleatoric_variance(y_pred):
    return F.softplus(y_pred[:, 0:1]) / F.softplus(y_pred[:, 1:2])


# model settings
LAYERS = [1, 64, 64, 1]
VAR_LAYERS = [1, 16, 2]  # gamma shape and rate
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]  # bounded
VAR_ACTIVATIONS = [nn.Tanh() for _ in range(len(VAR_LAYERS) - 2)]

# ------------------------------------ create data ------------------------------------
target = lambda x: torch.sin(x)
noise_std = lambda x: (x / 6 + 1) * NOISE_STD
noise = lambda x: torch.randn_like(x) * noise_std(x)  # mean 0

x_train1 = torch.rand(SAMPLES // 2) * (-3)
x_train2 = torch.rand(SAMPLES // 2) * 1.5 + 1.5
x_train = torch.cat([x_train1, x_train2], dim=0).reshape(-1, 1)
y_train = target(x_train) + noise(x_train)
x_train, y_train = x_train.to(device), y_train.to(device)

# -------------------------------- mean network training ------------------------------
mean_model = MLP(LAYERS, post_modules=ACTIVATIONS).to(device)
init_weights(mean_model, ACTIVATIONS[0])

tic = time.time()
mean_cost = train(mean_model, lambda: F.mse_loss(mean_model(x_train), y_train), EPOCHS)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------ variance network training ----------------------------
mean_model.eval()
with torch.no_grad():
    residual = (mean_model(x_train) - y_train) ** 2
residual = residual.clamp_min(1e-12)

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

# ------------------------------------ HMC sampling -----------------------------------
var_model.eval()
with torch.no_grad():
    var_train = aleatoric_variance(var_model(x_train))

model = MLP(LAYERS, post_modules=ACTIVATIONS).to(device)
model.load_state_dict(mean_model.state_dict())  # start the chain at the mean estimate
model.eval()

tau_params = [torch.tensor(1.0 / PRIOR_STD**2).to(device)] * len(
    list(model.parameters())
)
params_init = hamiltorch.util.flatten(model).to(device)

# heteroscedastic gaussian nll with the aleatoric variance held fixed; its log-variance
# term is constant in the weights and therefore does not enter the posterior
nll = lambda y_pred, y: 0.5 * (y - y_pred) ** 2 / var_train

tic = time.time()
params_hmc = hamiltorch.sample_model(
    model=model,
    x=x_train,
    y=y_train,
    params_init=params_init,
    model_loss=nll,
    num_samples=HMC_SAMPLES,
    step_size=HMC_STEP_SIZE,
    num_steps_per_sample=HMC_STEPS_PER_SAMPLE,
    tau_list=tau_params,
    verbose=True,
)
toc = time.time()
print(f"elapsed sampling time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-6, 6, 200).reshape(-1, 1).to(device)
pred_list = []

tic = time.time()
for i in range(BURN_IN, HMC_SAMPLES):
    torch.nn.utils.vector_to_parameters(params_hmc[i], model.parameters())
    with torch.no_grad():
        pred_list.append(model(x_test))
toc = time.time()
print(f"elapsed inference time {toc - tic:.2f} s")

y_preds = torch.stack(pred_list)
mean = y_preds.mean(dim=0)
epistemic_std = y_preds.std(dim=0)
with torch.no_grad():
    aleatoric_std = torch.sqrt(aleatoric_variance(var_model(x_test)))
total_std = torch.sqrt(aleatoric_std**2 + epistemic_std**2)

if not args.book:
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(mean_cost, "k")
    ax.plot(var_cost, "r")
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
        ax.set_ylim(-1.5, 1.5)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        CSV_DIR / "ale_hmc.csv",
        x=x_test.squeeze().cpu().numpy(),
        mean=mean.squeeze().cpu().numpy(),
        std_aleatoric=aleatoric_std.squeeze().cpu().numpy(),
        std_total=total_std.squeeze().cpu().numpy(),
        std_true_aleatoric=noise_std(x_test).squeeze().cpu().numpy(),
        mean_true=target(x_test).squeeze().cpu().numpy(),
    )
    save_csv(
        CSV_DIR / "ale_hmc_train.csv",
        x=x_train.squeeze().cpu().numpy(),
        y=y_train.squeeze().cpu().numpy(),
    )
