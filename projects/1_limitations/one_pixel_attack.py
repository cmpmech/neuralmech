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
from postprocessing import save_csv, show_image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 128

# define loss
cost_fun = nn.CrossEntropyLoss()

# model settings
LAYERS = [28 * 28, 256, 256, 10]
ACTIVATIONS = [nn.ReLU() for _ in range(len(LAYERS) - 2)]

# attack
SAMPLES = 1000  # test images searched
EXAMPLES = 4
UPSAMPLING = 10  # pixel blocks keep the exported images crisp
HIGHLIGHT = [0.773, 0.239, 0.255]  # changed pixel in col13 of the book palette

# ------------------------------------- load data -------------------------------------
mnist = load_dataset("ylecun/mnist").with_format("numpy")
X_train = torch.from_numpy(mnist["train"][:]["image"]).float() / 255
Y_train = torch.from_numpy(mnist["train"][:]["label"]).long()
X_test = torch.from_numpy(mnist["test"][:SAMPLES]["image"]).float() / 255
Y_test = torch.from_numpy(mnist["test"][:SAMPLES]["label"]).long()
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

# --------------------------- instantiate model & optimizer ---------------------------
model = nn.Sequential(nn.Flatten(), MLP(LAYERS, ACTIVATIONS)).to(device)
init_weights(model, ACTIVATIONS[0])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------------------- training --------------------------------------
tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    train_cost = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        cost = cost_fun(model(x), y)
        cost.backward()
        optimizer.step()
        train_cost += cost.item()
    pbar.set_postfix({"train": f"{train_cost / len(train_loader):.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# -------------------------------------- attack ---------------------------------------
model.eval()
flips = torch.arange(2 * 28 * 28, device=device)
rows, cols, values = flips // 2 // 28, flips // 2 % 28, (flips % 2).float()

results = []
with torch.no_grad():
    for idx in range(SAMPLES):
        x, y = X_test[idx].to(device), Y_test[idx].item()
        if model(x[None]).argmax().item() != y:
            continue
        # every pixel set to black and to white, one at a time
        candidates = x.expand(len(flips), -1, -1).clone()
        candidates[flips, rows, cols] = values
        prob = torch.softmax(model(candidates), dim=1)
        prob_true = prob[:, y]
        best = prob_true.argmin()
        pred = prob[best].argmax().item()
        if pred != y:
            results.append((idx, y, pred, prob[best, pred].item(), rows[best].item(),
                            cols[best].item(), values[best].item()))
correct = (model(X_test.to(device)).argmax(dim=1).cpu() == Y_test).sum().item()
print(f"{len(results)} of {correct} correctly classified test digits flip with a single pixel")

# most confident flips with distinct labels
examples = []
for result in sorted(results, key=lambda result: -result[3]):
    if result[1] not in [example[1] for example in examples]:
        examples.append(result)
examples = examples[:EXAMPLES]
for idx, y, pred, prob, row, col, value in examples:
    print(f"test digit {idx}: {y} -> {pred} {100 * prob:.2f}% by pixel ({row}, {col}) = {value:.0f}")

# ----------------------------------- postprocessing ----------------------------------
images = {}
for idx, y, pred, prob, row, col, value in examples:
    image = np.repeat(1 - X_test[idx].numpy()[:, :, None], 3, axis=2)  # white background
    images[f"{idx}"] = image
    attacked = image.copy()
    attacked[row, col] = HIGHLIGHT
    images[f"{idx}_attacked"] = attacked
images = {name: np.kron(image, np.ones((UPSAMPLING, UPSAMPLING, 1))) for name, image in images.items()}

if not args.book:
    fig, axs = plt.subplots(2, len(examples), figsize=(2 * len(examples), 4))
    for k, (idx, y, pred, prob, row, col, value) in enumerate(examples):
        axs[0, k].imshow(images[f"{idx}"])
        axs[1, k].imshow(images[f"{idx}_attacked"])
        axs[0, k].set_title(f"{y}")
        axs[1, k].set_title(f"{pred} {100 * prob:.1f}%")
    for ax in axs.flat:
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0, hspace=0.2)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, image in images.items():
        show_image(image, path=RGB_PDF_DIR / f"one_pixel_{name}.pdf", close=True)
    ids, labels, preds, probs, rows, cols, values = zip(*examples)
    save_csv(CSV_DIR / "one_pixel_attack.csv", idx=ids, label=labels, pred=preds, prob=probs,
             row=rows, col=cols, value=values)
