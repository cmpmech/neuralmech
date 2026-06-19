import argparse
import time
import warnings
from pathlib import Path

import hamiltorch
import matplotlib.pyplot as plt
import torch
from torch import nn

from NN import MLP
from postprocessing import save_csv

warnings.filterwarnings(
    "ignore", message=".*Converting a tensor with requires_grad=True to a scalar.*"
)

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
HMC_SAMPLES = 500
BURN_IN = 50  # samples to discard (remove bias from initialization)
HMC_STEP_SIZE = 0.002  # lower: more stable, higher: faster exploration
HMC_STEPS_PER_SAMPLE = 15
PRIOR_STD = 1.0
NOISE_STD = 0.14
SAMPLES = 32

# model settings
LAYERS = [1, 64, 64, 1]
ACTIVATIONS = [nn.Tanh() for _ in range(len(LAYERS) - 2)]  # bounded

# ------------------------------------ create data ------------------------------------
x_train1 = torch.rand(SAMPLES // 2) * (-3)
x_train2 = torch.rand(SAMPLES // 2) * 1.5 + 1.5
x_train = torch.cat([x_train1, x_train2], dim=0).reshape(-1, 1)
y_train = torch.sin(x_train) + torch.randn_like(x_train) * NOISE_STD
x_train, y_train = x_train.to(device), y_train.to(device)

# --------------------------------- instantiate model ---------------------------------
model = MLP(LAYERS, ACTIVATIONS).to(device)
model.eval()

tau_params = [torch.tensor(1.0 / PRIOR_STD**2).to(device)] * len(
    list(model.parameters())
)
tau_out = torch.tensor(1.0 / NOISE_STD**2).to(device)

# ------------------------------------ HMC sampling -----------------------------------
params_init = hamiltorch.util.flatten(model).to(device)

tic = time.time()
params_hmc = hamiltorch.sample_model(
    model=model,
    x=x_train,
    y=y_train,
    params_init=params_init,
    model_loss="regression",  # mse likelihood
    num_samples=HMC_SAMPLES,
    step_size=HMC_STEP_SIZE,
    num_steps_per_sample=HMC_STEPS_PER_SAMPLE,
    tau_out=tau_out,
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
mean = y_preds.mean(dim=0).cpu().squeeze()
std = y_preds.std(dim=0).cpu().squeeze()  # epistemic
total_std = torch.sqrt(std**2 + NOISE_STD**2)

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_train.cpu(), y_train.cpu(), "bo")
    ax.plot(x_test.squeeze().cpu(), mean, "k")
    ax.fill_between(
        x_test.squeeze().cpu(),
        (mean - 2 * total_std),
        (mean - 2 * std),
        alpha=0.3,
        color="b",
    )
    ax.fill_between(
        x_test.squeeze().cpu(),
        (mean + 2 * std),
        (mean + 2 * total_std),
        alpha=0.3,
        color="b",
    )
    ax.fill_between(
        x_test.squeeze().cpu(), (mean - 2 * std), (mean + 2 * std), alpha=0.3, color="r"
    )
    ax.set_ylim(-4, 4)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "hmc.csv",
        x=x_test.squeeze().cpu().numpy(),
        mean=mean.numpy(),
        std=std.numpy(),
        std_total=total_std.numpy(),
    )
    save_csv(
        RESULTS_DIR / "hmc_train.csv",
        x=x_train.squeeze().cpu().numpy(),
        y=y_train.squeeze().cpu().numpy(),
    )
