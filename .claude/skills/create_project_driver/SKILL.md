---
name: create_project_driver
description: Scaffold a new Python driver script in `code/projects/<N>_<name>/` following the NeuralMech driver conventions. Invoke when creating a new experiment script (training run, regression demo, optimizer comparison, ablation, etc.). Produces the canonical 10-step structure (imports → paths → device+seeds → argparse → constants → helpers → data → model → training → save) with the correct color palette, save formats, and `--book`/`--animate` plumbing. For library code or unit tests, use `create_unit_test`.
---

Create a new driver in `code/projects/<N>_<name>/`. This skill provides the full template; `python_hard_rules` covers the must-follow rules referenced throughout.

## Where the file goes

`code/projects/<N>_<name>/<descriptive_name>.py`, where `<N>` is the supporting book chapter (e.g., `3_mlp/`, `16_fwi/`). Drivers without a chapter (schematic figures, helpers) go under `0_*/`.

## Canonical template

Adapt this skeleton; do not reorder sections:

```python
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import MLP
from postprocessing import save_csv

# -------------------------------- paths --------------------------------
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
DATA_DIR = BASE_DIR / "../../data"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"

# ------------------------------- device --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# ------------------------------- argparse ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# ----------------------------- hyperparameters --------------------------
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 64
SAMPLES = 256
NOISE = 0.1
REGULARIZATION = 1e-4
PATIENCE = None

D_in = 1
H = 32
D_out = 1

# ------------------------------- helpers -------------------------------
f = lambda x: np.sin(2 * np.pi * x)
cost_fun = nn.MSELoss(reduction="mean")

# ----------------------------- data loading -----------------------------
x = np.linspace(-1, 1, SAMPLES).reshape(-1, 1)
y = f(x) + NOISE * np.random.randn(*x.shape)
X = torch.tensor(x, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, Y)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)

# ---------------------- instantiate model & optimizer -------------------
layers = [D_in, H, H, D_out]
activations = [nn.Tanh(), nn.Tanh(), None]

model = MLP(layers, activations).to(device)
init_weights(model, activations[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)

# ------------------------------- training -------------------------------
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
print_every = 10

tic = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = standardizex(x.to(device)), standardizey(y.to(device))
        optimizer.zero_grad()
        y_pred = model(x)
        cost = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = standardizex(x.to(device)), standardizey(y.to(device))
            y_pred = model(x)
            val_cost[epoch] += cost_fun(y_pred, y).item()
        val_cost[epoch] /= len(val_loader)

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}",
                          "val":   f"{val_cost[epoch]:.2e}"})

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------ postprocessing for book -----------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k", label="train")
ax.plot(val_cost, "r", label="val")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()

if args.book:
    save_csv(RESULTS_DIR / "loss_curve.csv",
             epoch=np.arange(len(train_cost)),
             train=train_cost,
             val=val_cost)
    fig.savefig(RESULTS_DIR / "loss_curve.pdf")
    plt.close()
elif args.animate:
    # frames go to ANIMATION_DIR
    plt.close()
else:
    plt.show()
```

## What to adapt

When the user describes the new driver, vary:
- Hyperparameters and architecture (layers, activations).
- Data generation (function `f`, sampling, noise).
- Loss function (MSE → cross-entropy, custom physics loss, etc.).
- Postprocessing (which CSVs and figures to save).

Do **not** vary:
- The 10-step section order.
- `Path(__file__).parent`, seed, deterministic flag.
- `--book` / `--animate` semantics.
- `:.2e` / `:.2f` formatting.

## Special cases

### CPU-only scripts (small matrices)

When the script runs faster on CPU (small networks, small data):

```python
device = torch.device("cpu")  # faster on cpu, because matrices are small
torch.manual_seed(0)
```

Skip `cudnn.deterministic` since there's no GPU path.

### Early stopping

Add when `PATIENCE` is set:

```python
import copy

best_val, epochs_since_improve, best_state = float("inf"), 0, None

# inside training loop, after val_cost[epoch] is computed:
if PATIENCE is not None:
    if val_cost[epoch] < best_val:
        best_val = val_cost[epoch]
        epochs_since_improve = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= PATIENCE:
            print(f"early stopping at epoch {epoch} (best val {best_val:.2e})")
            train_cost = train_cost[:epoch + 1]
            val_cost = val_cost[:epoch + 1]
            break
```

### Saving / loading models

```python
torch.save(model, BASE_DIR / "../../models/name.pt2")
# later:
model = torch.load("model.pt2", weights_only=False, map_location=device)
```

### Borderless image figures

For pixel-exact image output (e.g., reconstructions, loss landscapes):

```python
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis("off")
ax.set_rasterized(True)   # for PDF with contourf — avoids line artifacts
```

### Network types beyond MLP

Pull from `NN.py`:
- `DCN` for convolutional (1D/2D/3D) — pass `channels`, `activations`, `kernel_size`, `stride`, `padding`, `resamplings`, `normalizations`.
- `DGCN` / `DGSAGE` / `DGAT` / `DGIN` for graph nets — input is a PyG `Data` object.
- `DRNN` for recurrent — pass `cell` (RNN/LSTM/GRU), `final_activation`.
- `NODE` for neural ODE — wraps any `rhs_model`, call `forward(T, h0)`.
- `BayesianMLP` — call `.kl_divergence(prior_std)` for ELBO.
- `ResNet` — wraps an existing model with skip connections via `(start, end)` index pairs.
- `ELM` — call `.fit(x, y, regularization)` before forward.
- `AE` / `VAE` — `AE(Encoder, Decoder)`; VAE forward returns `(reconstruction, mean, logvar)`.
- `SIREN` — pass `omega_0`, init handles weight scaling.
- `KAN` — drop-in MLP alternative.

## Plotting palette (use consistently)

| color | meaning |
|---|---|
| `"k"` / `"black"` | training data, true function, primary prediction |
| `"r"` / `"red"` | validation data, uncertainty envelope |
| `"b"` / `"blue"` | reference lines, alternative model |
| `"cividis"` | loss landscape contours |
| `"hot_r"` (log-scaled) | error maps |

## File format rule

- Line/contour plots → `.pdf`.
- Raster/image plots → `.png`.

```python
fig.savefig(RESULTS_DIR / "name.pdf")   # line/contour
fig.savefig(RESULTS_DIR / "name.png")   # raster/image
```

## Output checklist

Before declaring the driver done:

- [ ] Sections in canonical order (10 steps).
- [ ] `BASE_DIR = Path(__file__).parent` near top.
- [ ] `torch.manual_seed` + `cudnn.deterministic` after device.
- [ ] `--book` / `--animate` plumbing matches the template.
- [ ] Constants in `ALL_CAPS`.
- [ ] No aligned `=`.
- [ ] Section headers `# ----- name -----` for each major step.
- [ ] `tqdm(range(EPOCHS))` with `pbar.set_postfix(...)` for live status.
- [ ] `tic`/`toc` wall-clock print at end of training.
- [ ] No docstrings, no `__main__` guard.
- [ ] Plot file format matches the rule (.pdf for line, .png for raster).
- [ ] CSVs via `save_csv(...)` from `postprocessing.py`, not manual writing.
