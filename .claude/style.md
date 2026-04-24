# NeuralMech — Python Driver Style Guide

This describes the conventions used throughout `projects/`. Emulate these patterns exactly when writing new drivers or extending existing ones.

---

## File structure (top to bottom)

Every driver follows this order. Do not reorder sections.

```
1. imports
2. BASE_DIR / RESULTS_DIR / DATA_DIR
3. device + random seeds
4. argparse (if --book / --animate needed)
5. hyperparameters (ALL_CAPS)
6. helper lambdas / loss functions
7. data generation or loading
8. model + optimizer instantiation
9. training loop
10. postprocessing / saving
```

---

## What not to do

- **No aligned assignments.** Never pad `=` signs to align a column of values — not for constants, not for dicts, not for anything. Use plain `x = 1` spacing throughout.

---

## Imports

Group in this order, blank line between each group:

```python
import argparse          # stdlib first
from pathlib import Path

import matplotlib.pyplot as plt   # third-party scientific
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from DL import Standardizer, init_weights   # local last
from NN import MLP
```

---

## Paths

```python
BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../../results"
DATA_DIR    = BASE_DIR / "../../data"
ANIMATION_DIR = RESULTS_DIR / "animations/animation_frames"
```

Always `Path(__file__).parent` — never `os.getcwd()` or hardcoded paths.

---

## Device and reproducibility

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
```

Seed goes immediately after device. Simple scripts that run on CPU only write:

```python
device = torch.device("cpu")  # faster on cpu, because matrices are small
```

---

## Argparse

Only `--book` and `--animate` flags; no positional arguments, no `--help` customisation.

```python
parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()
```

Behaviour:
- no flag → `plt.show()` (interactive preview)
- `--book` → write CSV / PDF to `RESULTS_DIR`, call `plt.close()`
- `--animate` → write frames to `ANIMATION_DIR`, call `plt.close()`

---

## Constants

All module-level config in `ALL_CAPS`:

```python
EPOCHS      = 1000
LR          = 1e-3
BATCH_SIZE  = 64
SAMPLES     = 256
NOISE       = 0.1
REGULARIZATION = 1e-4
PATIENCE    = None
```

Exception: in `15_generation`-style scripts (heavy GPU, no book output), lowercase config names are acceptable when they read more naturally (`epochs`, `lr`, `batch_size`).

---

## Section headers

Use this exact format everywhere — ~70-char total width, text centred with spaces:

```python
# ----------------------------- data loading -----------------------------
# ----------------------- instantiate model & optimizer ------------------
# ------------------------------- training -------------------------------
# ----------------------- postprocessing for book ------------------------
```

---

## Training loop

The canonical pattern. Copy it verbatim; adjust only what must change.

```python
train_cost = [0] * EPOCHS
val_cost   = [0] * EPOCHS
print_every = 10

tic  = time.time()
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = standardizex(x.to(device)), standardizey(y.to(device))
        optimizer.zero_grad()
        y_pred = model(x)
        cost   = cost_fun(y_pred, y)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)   # avg per batch

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
```

**Early stopping** (add when `PATIENCE` is set):

```python
best_val, epochs_since_improve, best_state = float("inf"), 0, None
...
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
                val_cost   = val_cost[:epoch + 1]
                break
```

---

## Data splitting

```python
dataset = TensorDataset(X, Y)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
Y_train = train_data.dataset.tensors[1][train_data.indices]
standardizex = Standardizer(X_train, dim=0)
standardizey = Standardizer(Y_train, dim=0)
```

---

## Model instantiation

```python
layers      = [D_in, H, H, D_out]
activations = [nn.Tanh(), nn.Tanh(), None]

model = MLP(layers, activations).to(device)
init_weights(model, activations[0])
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=REGULARIZATION)
```

`init_weights` from `DL.py` picks Kaiming or Xavier depending on the activation class — always call it after `.to(device)`.

---

## Saving outputs

**CSV for book figures** — via `save_csv` from `postprocessing.py`:

```python
from postprocessing import save_csv
save_csv(RESULTS_DIR / f"filename_{PARAM}.csv", x=x_arr, y=y_arr)
```

Keyword argument names become CSV column headers (space-separated).

**Model checkpoints**:

```python
torch.save(model, BASE_DIR / "../../models/name.pt2")
```

When loading: always `map_location=device` in `torch.load`, never `.to(device)` afterward.

---

## Plotting conventions

**Figure creation**: default `fig, ax = plt.subplots()`. Add `dpi=150` or `figsize` only when the output format requires it.

**Color semantics** (use consistently):
| color | meaning |
|-------|---------|
| `"k"` / `"black"` | training data, true function, primary prediction |
| `"r"` / `"red"` | validation data, uncertainty envelope |
| `"b"` / `"blue"` | reference lines, alternative model |
| `"cividis"` | loss landscape contours |
| `"hot_r"` (log-scaled) | error maps |

**Line / marker shorthands**: `"ko"` (black circles), `"ro"` (red circles), `"k--"` (dashed black), `"k:"` (dotted black).

**Metric formatting**: always `:.2e` for costs, `:.2f` for seconds.

**Saving**:
```python
fig.savefig(RESULTS_DIR / "name.pdf")   # line/contour plots → PDF
fig.savefig(RESULTS_DIR / "name.png")   # raster / image plots → PNG
plt.close()
```

Avoid tight-layout padding for borderless figures:
```python
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis("off")
ax.set_rasterized(True)   # PDF with contourf — avoids line artifacts
```

---

## Lambda functions

Preferred for simple true functions and numpy losses:

```python
f        = lambda x: np.sin(2 * np.pi * x)
cost_fun = lambda y_pred, y: np.mean((y - y_pred) ** 2)
```

For PyTorch losses, use `nn.MSELoss(reduction="mean")` directly.

---

## Comments

No docstrings in driver scripts. Only two comment forms are used:

1. **Section headers** (always, see above)
2. **Single inline comment** when the reason is non-obvious:
   ```python
   # faster on cpu, because matrices are small
   # 95 % confidence interval
   # do not penalize bias
   ```

Never write what the code does. Write why, only when surprising.

---

## No `if __name__ == "__main__"`

Drivers execute at module level. No main-guard, no `main()` function.

---

## Core library summary

### `NN.py` — architectures

| Class | Description |
|-------|-------------|
| `MLP` | Fully-connected; takes `layers`, `activations`, `normalizations`, `dropouts` |
| `DCN` | Conv net (1D/2D/3D); `channels`, `activations`, `kernel_size`, `stride`, `padding`, `resamplings`, `normalizations` |
| `DGCN`, `DGSAGE`, `DGAT`, `DGIN` | GNN variants; input is PyG `Data` (`graph.x`, `graph.edge_index`) |
| `DRNN` | RNN/LSTM/GRU; `layers`, `cell`, `final_activation`, `normalizations` |
| `NODE` | Neural ODE; wraps any `rhs_model`; call `forward(T, h0)` |
| `BayesianMLP` | Variational BNN; call `.kl_divergence(prior_std)` for ELBO |
| `ResNet` | Wraps existing model with skip connections via `(start, end)` index pairs |
| `ELM` | Random features + closed-form output layer; call `.fit(x, y, reg)` before forward |
| `AE` | Autoencoder; `AE(Encoder, Decoder)` |
| `VAE` | Variational AE; `forward` returns `(reconstruction, mean, logvar)` |
| `SIREN` | Sinusoidal net with `omega_0` frequency; init handles weight scaling |
| `KAN` | From `efficient_kan`; drop-in alternative to MLP |

### `DL.py` — training utilities

| Function/Class | Description |
|----------------|-------------|
| `init_weights(model, activation)` | Kaiming (ReLU family) or Xavier (Tanh/Sigmoid); zeros biases. Call after `.to(device)` |
| `Standardizer(X, dim)` | Zero-mean unit-variance; `__call__` normalises, `.inverse()` undoes |
| `Normalizer(X, dim)` | [0,1] range normalisation; same interface as Standardizer |
| `get_params / set_params / flatten_params / unflatten_params` | Landscape visualisation helpers |
| `filter_normalize_direction` | Filter-normalised random directions for loss landscape plots |
| `build_ae_cnn_config(depth, conv_layers, channel_dim, base)` | Returns `(channels, strides)` for symmetric AE CNN encoder |

### `postprocessing.py` — output helpers

| Function | Description |
|----------|-------------|
| `save_csv(path, **cols)` | Saves keyword args as space-separated CSV; column names = kwarg names |
| `show_image(img, grayscale, path, close)` | Displays/saves a numpy image without axes, pixel-exact sizing |
