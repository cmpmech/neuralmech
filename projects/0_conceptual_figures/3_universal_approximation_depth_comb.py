import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

neurons = 3

# ---------------------------- model settings ----------------------------
layers = [1, neurons, 1, neurons, 1]
activations = [
    torch.nn.ReLU(inplace=True),
    torch.nn.Identity(),
    torch.nn.ReLU(inplace=True),
]

# -------------------- instantiate model & optimizer ---------------------
model = MLP(layers, activations).to(device)

# -------------------- custom weight initialization ----------------------
with torch.no_grad():
    model.model[0].weight.data.fill_(1.0)
    for i in range(neurons):
        model.model[0].bias.data[i].fill_(1.0 - i * 2.0 / neurons)

    model.model[2].weight.data[0].fill_(neurons)
    model.model[2].bias.data.fill_(-1.0)
    for i in range(1, neurons):
        if i % 2 == 0:
            model.model[2].weight.data[:, i].fill_(2.0 * neurons)
        else:
            model.model[2].weight.data[:, i].fill_(-2.0 * neurons)

    model.model[4].weight.data.fill_(1.0)
    for i in range(neurons):
        model.model[4].bias.data[i].fill_(1.0 - (i - 0.3) * 2.0 / neurons)

    model.model[6].weight.data[0].fill_(-neurons)
    model.model[6].bias.data.fill_(1.0)
    for i in range(1, neurons):
        if i % 2 == 0:
            model.model[6].weight.data[:, i].fill_(-2.0 * neurons)
        else:
            model.model[6].weight.data[:, i].fill_(2.0 * neurons)

# ---------------------------- postprocessing ----------------------------
x_test = torch.linspace(-1, 1, 400).unsqueeze(1)
y_pred = model(x_test).detach()

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test, y_pred, "r")
    plt.show()

if args.book:
    save_csv(
        RESULTS_DIR / "universal_approx_depth_comb.csv",
        x=x_test.squeeze(),
        ypred=y_pred.squeeze(),
    )
