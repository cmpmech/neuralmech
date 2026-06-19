import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

# -------------------------------------- settings -------------------------------------
NEURONS = 20

# model settings
LAYERS = [1, NEURONS, 1]
ACTIVATIONS = [nn.ReLU(inplace=True)]

# ----------------------------------- training data -----------------------------------
SAMPLES = NEURONS + 1
x_train = torch.linspace(-1, 1, SAMPLES).unsqueeze(1)
y_train = torch.sin(torch.pi * x_train)

# --------------------------------- instantiate model ---------------------------------
model = MLP(LAYERS, ACTIVATIONS)
model.to(device)

# ---------------------------- custom weight initialization ---------------------------
# spread the hidden ReLU kinks evenly across the input range
with torch.no_grad():
    model.model[0].weight.data.fill_(1.0)
    for i in range(NEURONS):
        model.model[0].bias.data[i].fill_(1.0 - i * 2.0 / NEURONS)
    model.model[2].weight.data.fill_(1.0)
    model.model[2].bias.data.fill_(0.0)


# ---------------------------------- fit output layer ---------------------------------
# the output layer is linear in the hidden features, so least squares solves it exactly
def predict_hidden(x):
    x = model.model[0](x)
    x = model.model[1](x)
    return x


with torch.no_grad():
    X = torch.hstack([predict_hidden(x_train), torch.ones(SAMPLES, 1)])
    fit = torch.linalg.inv(X.T @ X) @ X.T @ y_train
    model.model[2].weight.data[0, :] = fit[0:NEURONS, 0]
    model.model[2].bias.data[:] = fit[NEURONS, 0].item()

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-1, 1, 400).unsqueeze(1)
y_test = torch.sin(torch.pi * x_test)
y_pred = model(x_test).detach()

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, "bo")
    ax.plot(x_test, y_test, "k")
    ax.plot(x_test, y_pred, "r")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / f"universal_approx_width{NEURONS}.csv",
        x=x_test.squeeze(),
        y=y_test.squeeze(),
        ypred=y_pred.squeeze(),
    )
