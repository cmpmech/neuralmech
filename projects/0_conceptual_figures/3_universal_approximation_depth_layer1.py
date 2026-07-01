import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)

# -------------------------------------- settings -------------------------------------
NEURONS = 3

layers = [1, NEURONS, 1]
activations = [torch.nn.ReLU(inplace=True)]

# --------------------------------- instantiate model ---------------------------------
model = MLP(layers, post_modules=activations)

# initialize weights
with torch.no_grad():
    model.model[0].weight.data.fill_(1.0)
    for i in range(NEURONS):
        model.model[0].bias.data[i].fill_(1.0 - i * 2.0 / NEURONS)

    model.model[2].weight.data[0].fill_(NEURONS)
    model.model[2].bias.data.fill_(-1.0)
    for i in range(1, NEURONS):
        if i % 2 == 0:
            model.model[2].weight.data[:, i].fill_(2.0 * NEURONS)
        else:
            model.model[2].weight.data[:, i].fill_(-2.0 * NEURONS)

# ----------------------------------- postprocessing ----------------------------------
x_test = torch.linspace(-1, 1, 400).unsqueeze(1)
y_pred = model(x_test).detach()

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test, y_pred, "r")
    plt.show()

if args.book:
    save_csv(
        CSV_DIR / "universal_approx_depth_1.csv",
        x=x_test.squeeze(),
        ypred=y_pred.squeeze(),
    )
