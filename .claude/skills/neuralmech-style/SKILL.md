---
name: neuralmech-style
description: The authoritative Python rules and house style for the NeuralMech `code/` repo. Covers the must-follow hard rules (paths via Path(__file__), torch seeds, model loading with map_location, import order, no aligned `=`) and the full driver style (87-char banners and vocabulary, bare scaffolding, resolved paths, ALL_CAPS constants, the 10-step structure, no plot legends, subplots_adjust, comment/print discipline, README shape, chapter/naming registries), plus a report-only audit mode. Invoke when writing or editing ANY `.py` in `code/` (drivers, library, helpers), cleaning a file to match style, or auditing a project.
---

The authoritative Python rules and house style for the book's reference code. The
guiding value is **simplicity that reads fast**. When in doubt about whether a
simplification is faithful, ask before applying it.

The **Hard rules** below are must-follow (violations are defects); the rest is
house style. To audit a project without editing, see **Audit mode** at the end.

## Hard rules (must-follow)

- **Paths** via `BASE_DIR = Path(__file__).parent`; never `os.getcwd()` or absolute
  paths. Directory constants resolved (see Paths).
- **Seeds**: `torch.manual_seed(...)` + `torch.backends.cudnn.deterministic = True`
  right after device setup (see Device & reproducibility).
- **Model loading**: `torch.load(path, weights_only=False, map_location=device)` —
  never `.to(device)` after load (it silently re-moves a CPU-loaded model and can
  OOM on a small GPU).
- **Imports** grouped stdlib / third-party / local, one blank line between (see
  Imports).
- **No aligned `=`** anywhere — single space each side.
- **Constants**: module-level config in `ALL_CAPS` (see Constants).
- **No `__main__` guard** and **no docstrings** in drivers; they run top-to-bottom.
- **No plot legends** (`ax.legend()` / `label=`) — see Postprocessing.
- **`init_weights(model, activation)` after `.to(device)`** — never before.

Details and the rest of the style follow.

Ground-truth examples to imitate: the hand-cleaned `projects/3_sine/mlp_sine_classification.py`,
`projects/3_sine/discrete_sine_gen.py`, `projects/3_sine/README.md`, and the
gold-standard `projects/0_conceptual_figures/5_wasserstein.py`.

## Simplicity first

- Prefer the simplest readable form. Use intermediate variables instead of nested
  calls:

  ```python
  mlp_input = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)   # yes
  # not: torch.cat((x.flatten().unsqueeze(1), y.flatten().unsqueeze(1)), 1)

  x = torch.linspace(-1, 1, SAMPLES)
  y = torch.linspace(-1, 1, SAMPLES)
  x, y = torch.meshgrid(x, y, indexing="ij")     # bind on its own lines
  ```

- Don't sprinkle `.to(device)` when `device` is CPU anyway.
- Lean on the `ponytail` / `simplify` plugins to cut bloat.

## Imports

Three groups, one blank line between, in order: stdlib, third-party, local
(repo modules `DL`/`NN`/`postprocessing`/… and the project-local `helper`). No
imports after the first non-import statement; no unused imports.

```python
import argparse                  # stdlib
from pathlib import Path

import matplotlib.pyplot as plt  # third-party
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import Standardizer, init_weights   # local
from NN import MLP
```

Use `from torch import nn` (the repo's de-facto form), not `import torch.nn as nn`.

## Banners

- **Exactly 87 characters wide**, centered label, comment dashes. Build as
  `"# " + "-"*left + f" {label} " + "-"*right` where `left + right = 87 - 2 - len(label) - 2`
  and `left = ceil`, `right = floor` of the dash budget. Example (87 chars):
  ```
  # -------------------------------------- settings -------------------------------------
  ```
- **Flush-left at column 0, always** — even inside an `if`/`for`/`else` block; never
  indent a `# ----` line.
- **Controlled vocabulary** (verb/noun, reused across files — keep the set small):
  `settings`, `prepare data`, `create data`, `load data`, `load image`,
  `preprocessing`, `instantiate model & optimizer`, `training`, `postprocessing`,
  `book postprocessing`, `export`, `setup`, `helper`. At most 1–2 file-specific
  custom banners are fine and need not join this list.
- **No banners on scaffolding.** Imports, `BASE_DIR`/dir constants, device+seeds,
  and argparse all sit bare. The **first** banner in a driver is `settings`.

### `settings` sub-structure

Inside the single `settings` banner, group with these fixed lowercase
sub-comments (only those that apply, in this order):

```python
# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = ...
LR = ...
BATCH_SIZE = ...

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [...]
ACTIVATIONS = [...]
```

A file-specific sub-group comment (e.g. `# interpolation`) is fine when it earns
its keep. The old separate `training settings` / `model settings` /
`interpolation settings` banners are merged into this one `settings` block.

## Paths

- `BASE_DIR = Path(__file__).parent` (never `os.getcwd()` or absolute paths).
- Directory constants are **resolved**: `RESULTS_DIR = (BASE_DIR / "../../results").resolve()`;
  likewise `DATA_DIR`, `ANIMATION_DIR`.
- Book outputs go to fixed subfolders, never the `results/` root: figures to
  `RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()` (`savefig(RGB_PDF_DIR / ...)`),
  plot data to `CSV_DIR = (RESULTS_DIR / "data").resolve()` (`save_csv`/`savetxt`).
  These two folders are committed (`.gitkeep`) and **assumed to exist** — do not
  `mkdir` them.
- Naming: `*_DIR` for directories, plain names for files. Specialized subdirs that are
  *not* committed (e.g. per-case animation-frame folders) must be ensured to exist:
  `dir.mkdir(parents=True, exist_ok=True)`.

## Constants

- All fixed module-level config in `ALL_CAPS` (`EPOCHS`, `LR`, `BATCH_SIZE`,
  `LAYERS`, `ACTIVATIONS`, `CLASSES`, ...).
- **Minimize how many there are.** Consolidate or drop settings; prefer a shared
  vocabulary across drivers (`N`, `SAMPLES`, `RESOLUTION`, `REGULARIZATION`, ...).
  When a driver carries many knobs, ask which are genuinely fixed-but-present
  versus removable before adding them all.
- No aligned `=` columns — single space each side, everywhere.
- Exception: heavy GPU scripts (`15_generation`-style) may use lowercase config
  names where they read more naturally.

## Device & reproducibility

- Keep the **full** device/seed setup verbatim, even in CPU-only drivers — it is
  pedagogical (the book shows readers how to set it all up):
  ```python
  torch.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  device = torch.device("cpu")  # faster on cpu, because matrices are small
  ```
  Likewise, **keep** the `.to(device)` / `.cpu()` calls in the loops and at
  inference even when `device` is CPU — do not strip them. (This overrides the
  "avoid `.to(device)` on CPU" line in the notes: in drivers the explicit setup is
  the teaching point.) The "no redundant `.to(device)`" simplicity guidance still
  applies to genuinely throwaway internal code, not the driver device plumbing.
- Don't churn `torch.manual_seed` values in files whose saved book results depend
  on the exact seed.
- New NumPy randomness: `rng = np.random.default_rng(2)` style — but leave existing
  `np.random.seed(...)` files alone if their results depend on it.

## Activations

Always a comprehension, one fresh instance per layer:

```python
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]
```

Convert any `[Act()] * N` to this form when cleaning.

## Early stopping

When a driver exposes a `PATIENCE` setting (`None` to disable), track the best
validation state and stop once it stops improving. Restore the best weights after
the loop. Reference: `projects/3_sine/mlp_sine_regularization.py`.

```python
import copy

best_val = float("inf")
epochs_since_improve = 0
best_state = None
# ... inside the epoch loop, after val_cost[epoch] is computed:
    if PATIENCE is not None:
        if val_cost[epoch] < best_val:
            best_val = val_cost[epoch]
            epochs_since_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"early stopping at epoch {epoch} (best val {best_val:.2e})")
                train_cost = train_cost[: epoch + 1]
                val_cost = val_cost[: epoch + 1]
                break

if PATIENCE is not None and best_state is not None:
    model.load_state_dict(best_state)
```

## Postprocessing & figures

- Use `fig.subplots_adjust(left=0, right=1, top=1, bottom=0)` — never
  `fig.tight_layout()` or bbox tricks inside `savefig`.
- File formats: book figures export `.pdf` (vector plots stay vector; raster fields
  carry `set_rasterized(True)` so the pdf embeds an image — `imshow` needs no flag, it
  is always an embedded image), saved into `RGB_PDF_DIR`. `.jpg` for animation frames.
  `projects/0_cmyk_export/figures_to_cmyk.py` then derives the CMYK print set the book
  imports from `rgb_pdf/`/`rgb_png/` (see that project's README).
- **No legends.** Never `ax.legend()` or `label=` — the book renders legends in
  TikZ. Keep the color convention instead (`"k"` train/true, `"r"` val/uncertainty,
  `"b"` reference/alt).
- `--book` writes figures to `RGB_PDF_DIR` and data/CSVs to `CSV_DIR` under a
  `book postprocessing` banner; no flag → interactive `plt.show()`. Typical shape:

  ```python
  # ----------------------------------- postprocessing ----------------------------------
  ...build arrays/preds...

  if not args.book:
      ...plt.show() previews...
  # -------------------------------- book postprocessing --------------------------------
  else:
      plt.savefig(RGB_PDF_DIR / "name.pdf")
      save_csv(CSV_DIR / "name.csv", ...)
  ```

- Colormaps (fixed per domain): concepts `cividis`; diverging concepts `Spectral`;
  binary masks `binary`; abs/squared errors `hot_r` **always log-scaled**;
  elasticity `turbo`; waves `seismic`; heat `inferno`/`magma` (`cmasher.torch` for
  topology-optimized temperature fields); stresses/strains and other derived
  quantities `rainbow_desaturated` (replaces the former `cmasher.pride`).
  Non-builtin maps load from `.cmap/` via `load_cmap`.

## Comments & prints

- Minimal. Explain **WHY**, not WHAT — never restate what a name or general
  knowledge already says.
- **ASCII only** everywhere (code and READMEs): no greek letters, no arrows/unicode.
- Lowercase in comments and `print` strings, except real identifiers/abbreviations.
- Prints are rare — elapsed time (`print(f"elapsed time {toc - tic:.2f} s")`),
  or save locations in export/data-gen scripts.

## argparse

- Add `--book` / `--animate` only when they're meaningful for that driver.
- If animation is plausible but not yet implemented, leave
  `# TODO could be animated` on the relevant `parser.add_argument` (see
  `image_reconstruction_modl.py`).

## Driver structure

Drivers run top-to-bottom in this order — do not reorder:

1. imports
2. `BASE_DIR` / `RESULTS_DIR` / `DATA_DIR`
3. device + seeds
4. argparse (if `--book` / `--animate` needed)
5. `settings` (hyperparameters, loss, model settings — `ALL_CAPS`)
6. helper lambdas / functions
7. data generation or loading
8. model + optimizer instantiation
9. training loop
10. postprocessing / saving

No `__main__` guard, no docstrings (drivers only). `init_weights(model, activation)`
is called **after** `.to(device)`. Output formatting: costs `:.2e`, seconds `:.2f`
(`f"elapsed time {toc - tic:.2f} s"`).

## Training loop

Canonical shape (see `projects/3_sine/mlp_sine_regression.py`): `tqdm(range(EPOCHS))`
with `pbar.set_postfix`; `model.train()` / `model.eval()` toggled; validation under
`with torch.no_grad():`; `optimizer.zero_grad()` each step; accumulate `cost.item()`
and divide by `len(loader)` per epoch; `tic`/`toc` around the loop with a final
elapsed-time print.

## Structure & shared code

- Project-local helpers live in a file named exactly `helper.py`.
- Promote shared code to `NN.py` / `DL.py` / `ML.py` / `postprocessing.py` /
  `solvers` / `optimization.py` — but **ask the author first** before moving a
  helper to a shared module.
- Library/shared code uses docstrings + argument type hints; drivers do not (and
  drivers have no `__main__` guard — they run top-to-bottom).
- Function names: verb-noun by default. Naming registry (recurring banner labels,
  constant names, and local-variable names) is below — reuse from it before
  inventing a new name.

## README shape

- **Intro = shared context only, kept short (1–3 sentences).** Point to the book
  chapter and frame the *common thread* across the drivers — the running target,
  the shared test functions, the dataset, what unifies the project. Do **not**
  restate or enumerate what individual drivers do: the per-driver lines already
  carry that, and an intro that walks through each driver's topic is redundant.
  When in doubt, cut it back (e.g. `3_mlp`: a single line "Toy problems for
  Chapter 3 ... covering multilayer perceptron fundamentals", not a sentence
  naming forward prop, backprop, autodiff, expressivity, and universal
  approximation in turn).
- One line per driver saying what it investigates; one line per data-gen script
  (`script.py -> data/x.npz`), with a `_needs x.npz_` note. Omit cross-cutting
  facts (the `--book`/`--animate` flags, that drivers read `.npz`, that data-gen
  runs first).
- ASCII only; LaTeX math is allowed for explanations.
- Optional non-obvious-technicalities section: when the author tags a snippet
  `@claude`, add a short book-style explanation (essential math/code only). Title
  the section exactly **`## Non-obvious technicalities (authored by Claude)`** — the
  `(authored by Claude)` bracket marks this prose as machine-authored so the author
  knows to review it. Use it on any section you author in a README.
- Templates to match: `projects/3_sine/README.md` (basic), `projects/1_3D_imaging/README.md`
  (with an authored technicalities section).

## Naming registry

The single source of truth for recurring names. **Before introducing a name, reuse
one from here.** When a genuinely new recurring concept appears, add it here so the
next driver stays consistent. Grow this list as more projects are cleaned.

### Banner labels

The controlled vocabulary is listed under **Banners** above:
`settings`, `prepare data`, `create data`, `load data`, `load image`,
`preprocessing`, `instantiate model & optimizer`, `training`, `postprocessing`,
`book postprocessing`, `export`, `setup`, `helper`. Plus `settings` sub-comments
`# hyperparameters`, `# define loss`, `# model settings`. At most 1–2 file-specific
custom banners per file.

### Constant names (`ALL_CAPS`, settings block)

| name | meaning |
|---|---|
| `EPOCHS` | training epochs |
| `LR` | learning rate |
| `BATCH_SIZE` | minibatch size |
| `REGULARIZATION` | weight-decay / penalty strength |
| `DROPOUT` | dropout probability |
| `PATIENCE` | early-stopping patience (or `None`) |
| `SAMPLES` | number of data samples |
| `NOISE` | additive-noise amplitude |
| `CLASSES` | number of classes |
| `LAYERS` | MLP layer-width list |
| `ACTIVATIONS` | per-layer activation list (comprehension) |
| `HIDDEN_LAYERS` | hidden-layer count (when widths are built from it) |
| `RESOLUTION` | side length of a square sampling grid or raster image |
| `ITERS` | iteration count of a solver/loop (qualified counts keep the `_ITER` suffix, e.g. `ADMM_ITER`, `CG_ITER`) |
| `N` | generic count when nothing more specific fits |

Settled conventions:
- **Counts are bare nouns**, no `N_` prefix (`SAMPLES`, `CLASSES`, `EPOCHS`,
  `HIDDEN_LAYERS`) — `N` only as the generic fallback.
- **`RESOLUTION`** is the canonical name for grid/image size (not `IMG_SIZE`).

Driver-specific knobs already in use: `ENSEMBLE_SAMPLES`, `GRID_STEPS`,
`ALPHA_RANGE`, `DATA_HALF`.

### Local variable names (drivers)

| name | meaning |
|---|---|
| `cost_fun` | loss function |
| `model`, `optimizer` | the network and its optimizer |
| `dataset`, `train_data`, `val_data` | full dataset and the split subsets |
| `train_loader`, `val_loader` | the `DataLoader`s |
| `standardizex`, `standardizey` | input/output `Standardizer`s |
| `X_train`, `Y_train`, `X_val`, `Y_val` | full-array tensors (uppercase `X`/`Y`) |
| `x`, `y` | a minibatch (lowercase) inside the loop |
| `y_pred` | model output on a batch |
| `x_test`, `y_test`, `y_pred_test` | evaluation grid and predictions on it |
| `train_cost`, `val_cost` | per-epoch loss-history lists |
| `tic`, `toc` | wall-clock timing |
| `pbar`, `print_every`, `epoch` | training-loop progress |
| `idx` / `ids` | single index / collection of indices |

Convention: uppercase `X`/`Y` for full dataset arrays and split subsets; lowercase
`x`/`y` for minibatches and the evaluation grid.

## Book chapter reference

**Do not open or grep the book LaTeX by default.** Cleaning, writing, or auditing a
`.py` is a code-only task — the chapter-title table below is all you need for a
README's chapter line, and you must not read `../book/**` to do it. Only consult the
book when the author **explicitly asks to make the cross-repo connection** (e.g.
"does the code match the book?", "check the chapter's loss formula", "name the
exact section"). When in doubt, stay in `code/` and ask.

When the author does ask for the connection: the book lives in the sibling repo
`../book` (one level out of `code/`). Each `projects/<N>_*/` maps to **book chapter
`<N>`**, source at `../book/content/chapters/chapter<N>.tex`. A README's chapter
line should name the real chapter title; cite the specific `\section`/`\subsection`
when the project only supports part of a chapter (e.g. `1_3D_imaging` → the "3D
Medical Image Reconstruction" subsection of chapter 1). Don't deep-read the book —
grep the chapter `.tex` for the section heading and the `Generated by
\gitfile{...}` caption that names the driver. Chapter titles:

| # | title | # | title |
|---|---|---|---|
| 1 | Computational Mechanics Meets Artificial Intelligence | 11 | Neural Surrogates |
| 2 | Fundamental Machine Learning | 12 | Neural Solvers |
| 3 | Artificial Neural Networks | 13 | Physics-Informed Neural Networks |
| 4 | Neural Network Architectures | 14 | Constitutive Modeling with Neural Networks |
| 5 | Probabilistic Deep Learning | 15 | Generative Artificial Intelligence |
| 6 | Machine Learning Algorithms | 16 | Neural Optimization |
| 7 | Practical Machine Learning | 17 | Large Language Models |
| 8 | Governing Equations | 18 | Simulation Acceleration via GPUs |
| 9 | Numerical Methods | 19 | Deep Reinforcement Learning |
| 10 | Machine Learning in Computational Mechanics | 20 | Computational Mechanics After Artificial Intelligence |

The README's `(parenthetical)` after the chapter number may be a short topic gloss
rather than the exact title (e.g. `3_sine` uses "Multilayer Perceptrons" though
chapter 3 is "Artificial Neural Networks") — prefer the real title, but a clear
gloss is acceptable if one is already in use.

## Tracked consistency ledger

Decisions enforced uniformly across drivers — keep this list current as new ones
are settled:

- **Imports**: `import torch` then `from torch import nn` (the de-facto form in
  every `3_sine` driver). Do **not** rewrite to `import torch.nn as nn`.
- **`settings` sub-comments**: `# hyperparameters`, `# define loss`,
  `# model settings` (see above).
- **f-strings without placeholders**: drop the `f` prefix —
  `CSV_DIR / "name.csv"`, not `CSV_DIR / f"name.csv"`. Keep `f` only when
  the path actually interpolates (`f"name_{EPOCHS}.csv"`).
- **`None` comparisons**: `x is None` / `x is not None`, never `== None`.
- **Index names**: `idx` (single index) / `ids` (collection), not `indices` or
  `index`. Avoid bare `id` — it shadows the Python builtin.
- **Fix typos** in comments/identifiers when cleaning (don't replicate them).
- **Don't add `drop_last=True`** when it would empty a small single-batch loader.

## Audit mode (report-only)

When asked to **audit / review / unify style** for a project (e.g. "audit
projects/3_mlp") rather than edit: read every `.py` in `projects/<N>_*/` (skip
`__pycache__`, `README.md`), scan against this skill, and report a punch list with
`file:line` and the suggested fix — grouped by topic, ending with a summary line.
**Do not auto-edit**; surfacing the list is the deliverable. Cover at least: paths
& seeds, model loading, import grouping, the 10-step order, aligned `=`,
`ALL_CAPS` constants, banners (87-char, vocabulary, none on scaffolding), argparse
flags, the training-loop checklist, plotting (palette, no legends, book figures save
`.pdf` with `set_rasterized` on raster fields, `:.2e`/`:.2f`), comments/docstrings, `__main__`
guard, `init_weights` order, and `save_csv` usage. This is style enforcement, not
code review — don't critique comment wording or variable choices beyond the
registries.

```
## paths & reproducibility
- mlp_expressivity.py:14 — missing torch.manual_seed(...) after device line
- mlp_from_scratch.py:67 — torch.load(...).to(device) -> use map_location=device

## aligned assignments
- mlp_expressivity.py:23-28 — aligned `=` block, collapse to single space

Summary: 7 findings across 4 files. No auto-fix applied.
```

## Note on the mlhp skills

`mlhp-cpp-style` / `mlhp-overview` are useful only as _secondary_ inspiration for
how a style skill reads. Those conventions are C++/mlhp-specific and do not govern
the Python drivers.
