import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import init_weights
from NN import MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 128

# define loss
cost_fun = nn.CrossEntropyLoss()

# model settings
LAYERS = [28 * 28, 512, 512, 10]
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]

# memorization
SAMPLES = 10000  # training subset, keeps the memorization fast

# ------------------------------------- load data -------------------------------------
rng = np.random.default_rng(0)
mnist = load_dataset("ylecun/mnist").with_format("numpy")
X_train = torch.from_numpy(mnist["train"][:SAMPLES]["image"]).float() / 255
Y_train = torch.from_numpy(mnist["train"][:SAMPLES]["label"]).long()
X_test = torch.from_numpy(mnist["test"][:]["image"]).float() / 255
Y_test = torch.from_numpy(mnist["test"][:]["label"]).long()
labels = {"true": Y_train, "random": torch.from_numpy(rng.integers(0, 10, SAMPLES))}


# -------------------------------------- helper ---------------------------------------
def accuracy(X, Y):
    with torch.no_grad():
        Y_pred = model(X.to(device)).argmax(dim=1).cpu()
    return (Y_pred == Y).float().mean().item()


# ------------------------------------- training --------------------------------------
history = {}
tic = time.time()
for name, Y in labels.items():
    model = nn.Sequential(nn.Flatten(), MLP(LAYERS, ACTIVATIONS)).to(device)
    init_weights(model, ACTIVATIONS[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader = DataLoader(TensorDataset(X_train, Y), batch_size=BATCH_SIZE, shuffle=True)

    train_acc, test_acc = [], []
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            cost = cost_fun(model(x), y)
            cost.backward()
            optimizer.step()
        model.eval()
        train_acc.append(accuracy(X_train, Y))
        test_acc.append(accuracy(X_test, Y_test))
        pbar.set_postfix({"train": f"{train_acc[-1]:.3f}", "test": f"{test_acc[-1]:.3f}"})
    history[name] = (np.array(train_acc), np.array(test_acc))
    print(f"{name} labels: train accuracy {100 * train_acc[-1]:.2f}%, test accuracy {100 * test_acc[-1]:.2f}%")
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
epochs = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(6, 3))
for color, (train_acc, test_acc) in zip(["k", "b"], history.values()):
    ax.plot(epochs, train_acc, color)
    ax.plot(epochs, test_acc, color + "--")
fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.12)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "random_labels.csv", epoch=epochs,
             train_true=history["true"][0], test_true=history["true"][1],
             train_random=history["random"][0], test_random=history["random"][1])
plt.close()
