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
from NN import DCN, MLP
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
CHANNELS = [1, 32, 64, 10]

# translation
SHIFTS = np.arange(-8, 9)  # horizontal shift in pixels
EXAMPLE = 0  # test digit shown shifted
EXAMPLE_SHIFTS = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
UPSAMPLING = 10  # pixel blocks keep the exported images crisp

# ------------------------------------- load data -------------------------------------
mnist = load_dataset("ylecun/mnist").with_format("numpy")
X_train = torch.from_numpy(mnist["train"][:]["image"]).float()[:, None] / 255
Y_train = torch.from_numpy(mnist["train"][:]["label"]).long()
X_test = torch.from_numpy(mnist["test"][:]["image"]).float()[:, None] / 255
Y_test = torch.from_numpy(mnist["test"][:]["label"]).long()
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)


# -------------------------------------- helper ---------------------------------------
def shift(X, pixels):  # zero padded, digits never touch the border by more than a few pixels
    shifted = torch.zeros_like(X)
    if pixels >= 0:
        shifted[..., pixels:] = X[..., :X.shape[-1] - pixels]
    else:
        shifted[..., :pixels] = X[..., -pixels:]
    return shifted


# --------------------------- instantiate model & optimizer ---------------------------
activations = [nn.ReLU() for _ in range(len(LAYERS) - 2)]
mlp = nn.Sequential(nn.Flatten(), MLP(LAYERS, activations)).to(device)
init_weights(mlp, activations[0])

# the last convolution maps to the classes, global pooling makes the output shift invariant
cnn_activations = [[nn.ReLU(), nn.MaxPool2d(2)], [nn.ReLU(), nn.MaxPool2d(2)], None]
cnn = nn.Sequential(DCN(CHANNELS, cnn_activations, padding=1), nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()).to(device)
init_weights(cnn, nn.ReLU())
models = {"mlp": mlp, "cnn": cnn}

# ------------------------------------- training --------------------------------------
tic = time.time()
for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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

# ----------------------------------- postprocessing ----------------------------------
accuracy = {}
with torch.no_grad():
    for name, model in models.items():
        model.eval()
        accuracy[name] = []
        for pixels in SHIFTS:
            Y_pred = model(shift(X_test, pixels).to(device)).argmax(dim=1).cpu()
            accuracy[name].append((Y_pred == Y_test).float().mean().item())
        print(f"{name} accuracy at shifts {SHIFTS.tolist()}:",
              " ".join(f"{100 * acc:.0f}" for acc in accuracy[name]))

    x = X_test[EXAMPLE:EXAMPLE + 1]
    examples = [shift(x, pixels) for pixels in EXAMPLE_SHIFTS]
    for name, model in models.items():
        preds = [model(example.to(device)).argmax().item() for example in examples]
        print(f"{name} predictions of test digit {EXAMPLE} ({Y_test[EXAMPLE]}) at shifts {EXAMPLE_SHIFTS}: {preds}")
images = [np.kron(1 - example[0, 0].numpy(), np.ones((UPSAMPLING, UPSAMPLING))) for example in examples]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(SHIFTS, accuracy["mlp"], "k")
ax.plot(SHIFTS, accuracy["cnn"], "b")
fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.12)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "translation_shift.csv", shift=SHIFTS, mlp=accuracy["mlp"], cnn=accuracy["cnn"])
    for pixels, image in zip(EXAMPLE_SHIFTS, images):
        show_image(image, grayscale=True, path=RGB_PDF_DIR / f"translation_{pixels:+d}.pdf", close=True)
plt.close()
