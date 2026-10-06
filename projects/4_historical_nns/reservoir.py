import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small
torch.set_default_dtype(torch.float64)

# -------------------------------------- settings -------------------------------------
# hyperparameters
REGULARIZATION = 1e-6  # ridge penalty on the readout
WASHOUT = 50  # initial states discarded before fitting the readout

# model settings
RESOLUTION = 512
HIDDEN = 400
SPECTRAL_RADIUS = 0.9  # below one for the echo state property
INPUT_SCALING = 0.5

# ------------------------------------ create data ------------------------------------
t = torch.linspace(0, 2, RESOLUTION, device=device)
y = 0.5 * (torch.sin(10 * torch.pi * t) + torch.sin(23.7 * torch.pi * t))
TRAIN = RESOLUTION // 2

# --------------------------- instantiate model & optimizer ---------------------------
W_x = INPUT_SCALING * (2 * torch.rand(1, HIDDEN, device=device) - 1)
W_h = 2 * torch.rand(HIDDEN, HIDDEN, device=device) - 1
W_h *= SPECTRAL_RADIUS / torch.linalg.eigvals(W_h).abs().max()

# -------------------------------------- training -------------------------------------
tic = time.time()
h = torch.zeros(1, HIDDEN, device=device)
states = []
for tau in range(1, TRAIN):
    h = torch.tanh(y[tau - 1].view(1, 1) @ W_x + h @ W_h)
    states.append(h)
H = torch.cat(states)[WASHOUT:]
Y = y[1:TRAIN, None][WASHOUT:]
W_y = torch.linalg.solve(H.T @ H + REGULARIZATION * torch.eye(HIDDEN), H.T @ Y)
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
x = y[TRAIN - 1].view(1, 1)
y_pred = []
for tau in range(TRAIN, RESOLUTION):
    h = torch.tanh(x @ W_x + h @ W_h)
    x = h @ W_y
    y_pred.append(x[0, 0])
y_pred = torch.stack(y_pred)
print(f"autonomous prediction cost {((y_pred - y[TRAIN:]) ** 2).mean():.2e}")

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(t.cpu(), y.cpu(), "k")
    ax.plot(t[TRAIN:].cpu(), y_pred.cpu(), "r--")
    ax.axvline(t[TRAIN].item(), color="b")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "reservoir_groundtruth.csv", t=t.cpu(), y=y.cpu())
    save_csv(CSV_DIR / "reservoir_train.csv", t=t[:TRAIN:8].cpu(), y=y[:TRAIN:8].cpu())
    save_csv(CSV_DIR / "reservoir_prediction.csv", t=t[TRAIN:].cpu(), ypred=y_pred.cpu())
