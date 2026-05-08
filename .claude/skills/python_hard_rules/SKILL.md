---
name: python_hard_rules
description: Must-follow Python rules for the NeuralMech code repo — paths via Path(__file__), torch reproducibility, model loading, import order, no aligned assignments. Invoke whenever editing or writing any `.py` file under `code/` (drivers in `projects/`, library code, helpers). These are hard rules; violations are defects.
---

Hard rules to follow on every Python edit in this repo. For a per-project style audit (full ruleset), invoke `unify_style`. For a new driver scaffold, invoke `create_project_driver`. For unit tests on library files, invoke `create_unit_test`. For metric-driven code improvements, invoke `optimize_code`.

## Paths

- **Always** `BASE_DIR = Path(__file__).parent` (with `from pathlib import Path`).
- **Never** `os.getcwd()` or hardcoded absolute paths.
- Construct sibling paths with `BASE_DIR / "../../results"` etc. — relative to the script, not the working directory.

## Reproducibility (every PyTorch script)

After device setup, immediately:

```python
torch.manual_seed(<seed>)
torch.backends.cudnn.deterministic = True
```

Both lines required. Use a fixed integer seed (e.g., `0`).

## Model loading

When loading a saved model, always pass `map_location=device` directly:

```python
model = torch.load("model.pt2", weights_only=False, map_location=device)   # correct
model = torch.load("model.pt2", weights_only=False).to(device)             # avoid
```

The `.to(device)` form silently moves the model after CPU-load and can fail on memory-constrained GPUs.

## Import order

Group with **one blank line between groups**, in this order:

```python
import argparse                  # stdlib first
from pathlib import Path

import matplotlib.pyplot as plt  # third-party scientific
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from DL import Standardizer, init_weights   # local last
from NN import MLP
```

## No aligned assignments

Never pad `=` signs to align a column of values. One space on each side, always:

```python
EPOCHS = 1000        # correct
LR = 1e-3
BATCH_SIZE = 64

EPOCHS      = 1000   # avoid (aligned)
LR          = 1e-3
BATCH_SIZE  = 64
```

Applies to constants, dicts, function defaults, attribute assignments — everywhere.

## Constants

Module-level configuration in **`ALL_CAPS`**:

```python
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 64
```

Exception: heavy GPU scripts (e.g., `15_generation`-style) may use lowercase config names when they read more naturally.

## Driver structure

Every driver follows this top-to-bottom order. Do not reorder:

1. imports
2. `BASE_DIR` / `RESULTS_DIR` / `DATA_DIR`
3. device + random seeds
4. argparse (if `--book` / `--animate` needed)
5. hyperparameters (`ALL_CAPS`)
6. helper lambdas / loss functions
7. data generation or loading
8. model + optimizer instantiation
9. training loop
10. postprocessing / saving

## No `__main__` guard

Drivers execute at module level. Never:

```python
if __name__ == "__main__":   # don't add this
    main()
```

## Driver flags (argparse)

Drivers accept exactly two flags, no more:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()
```

Behaviour:
- no flag → `plt.show()` (interactive preview)
- `--book` → write CSV/PDF to `RESULTS_DIR`, then `plt.close()`
- `--animate` → write frames to `ANIMATION_DIR`, then `plt.close()`

## Comments

- **No docstrings** in driver scripts.
- Only two comment forms allowed:
  1. **Section headers** (always): `# ----------------------- training -----------------------` (~70 char total, text centered with spaces).
  2. **Single inline comment** when the *reason* is non-obvious: `# faster on cpu, because matrices are small`.
- Never describe what the code does; only why, only when surprising.

## Init weights

Call `init_weights` from `DL.py` **after** `.to(device)`, with the activation as the second argument. Never before:

```python
model = MLP(layers, activations).to(device)
init_weights(model, activations[0])    # correct order
```

## Output formatting

- Costs: `:.2e` (e.g., `f"{cost:.2e}"`).
- Seconds: `:.2f` (e.g., `f"{toc - tic:.2f} s"`).
