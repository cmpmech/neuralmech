import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 1000  # 400
LR = 1e-2
REGULARIZATION = 0  # 1e0
BATCH_SIZE = 32
ENSEMBLE_SAMPLES = 100

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
# three hidden layers is sufficient (five to show overfitting)
LAYERS = [1, 24, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "sine.npz")
dataset = TensorDataset(
    torch.from_numpy(data["X"]).to(torch.float32),
    torch.from_numpy(data["Y"]).to(torch.float32),
)
train_data, _ = torch.utils.data.random_split(dataset, [0.5, 0.5])

# standardization
X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)


# -------------------------------------- ensembling -----------------------------------
x_test = torch.linspace(-1.3, 1.3, 200).unsqueeze(1)

y_preds = []
for _ in tqdm(range(ENSEMBLE_SAMPLES), desc="ensemble"):
# --------------------------- instantiate model & optimizer ---------------------------
    model = MLP(LAYERS, ACTIVATIONS)
    model.to(device)
    init_weights(model, ACTIVATIONS[0])
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------------- training -------------------------------------
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = standardizex(x), standardizey(y)
            optimizer.zero_grad()
            y_pred = model(x)
            cost_fun(y_pred, y).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(standardizex(x_test).to(device))
        y_pred_test = standardizey.inverse(y_pred_test).cpu().numpy()

    y_preds.append(y_pred_test.flatten())

# ----------------------------------- postprocessing ----------------------------------
y_preds = np.stack(y_preds)
y_pred_mean = np.mean(y_preds, 0)

if not args.book:
    fig, ax = plt.subplots()
    ax.plot(x_test[:, 0], y_preds.T, "k", alpha=0.2)
    ax.plot(x_test[:, 0], y_pred_mean, "k")
    plt.show()

# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(
        RESULTS_DIR / "mlp_sine_ensemble.csv",
        x=x_test[:, 0],
        ymean=y_pred_mean,
        **{f"y{i}": y_preds[i] for i in range(len(y_preds))},
    )
