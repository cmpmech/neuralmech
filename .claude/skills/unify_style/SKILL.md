---
name: unify_style
description: Per-project style audit for `code/projects/<N>_*/`. Walk the driver scripts in one project directory, report style violations against the NeuralMech driver conventions, and suggest fixes. Report-only — do not auto-apply changes. Invoke when explicitly asked to audit, review, or unify style for a project (e.g., "audit projects/3_mlp", "check style on the new driver"). For per-edit hard rules, use `python_hard_rules`.
---

Per-project style audit. Read every `.py` file in `code/projects/<N>_*/`, scan against the conventions below, and report a punch list with file:line references and suggested fixes.

**Report-only.** Do not auto-edit. Surfacing the list is the deliverable; the user reviews and applies.

For per-edit hard rules (Path, seeds, map_location, no aligned `=`, etc.), see `python_hard_rules`. This audit covers those plus the broader driver-style conventions.

## How to run

1. Identify the target directory from the user's request (e.g., `projects/3_mlp/`).
2. List `.py` files in it (excluding `__pycache__`, `README.md`, etc.).
3. Read each file fully.
4. For each section below, scan and collect findings.
5. Report grouped by section, with file:line.

## Audit sections

### 1. Paths and reproducibility

- `os.getcwd()` or hardcoded absolute path → flag, replace with `Path(__file__).parent`.
- Missing `BASE_DIR = Path(__file__).parent` near top → flag.
- Missing `torch.manual_seed(...)` after device setup → flag.
- Missing `torch.backends.cudnn.deterministic = True` → flag.
- `torch.load(...).to(device)` form → flag, switch to `map_location=device`.

### 2. File structure order

The 10-step order (imports → BASE_DIR → device+seeds → argparse → ALL_CAPS constants → helpers → data → model+optimizer → training → save). Flag any out-of-order section. Common violations:
- Constants defined after data generation.
- Argparse below the constants block.
- `import` statements scattered through the file rather than grouped at top.

### 3. Imports

- Three groups (stdlib, third-party, local) separated by exactly one blank line each → flag missing blanks or group mixing.
- `import` after the first non-import statement → flag.
- Unused imports → flag.

### 4. Aligned assignments

Grep for any `\s+=\s` pattern with extra padding. Flag every aligned-`=` block:

```python
EPOCHS      = 1000   # flag
LR          = 1e-3
```

Suggest: collapse to single-space `=`.

### 5. Constants

- Module-level config not in `ALL_CAPS` → flag (unless this is a `15_generation`-style script where lowercase is acceptable).
- Constants defined after they are first used → flag.

### 6. Argparse

- More than `--book` and `--animate` flags → flag.
- Missing `--book` / `--animate` plumbing while `RESULTS_DIR` is written → flag.
- `--help` customization → flag, remove.

### 7. Section headers

- Major step (data, training, postprocessing, model setup) without a section header line → flag.
- Header not following `# ----- name -----` pattern (~70 chars, text centered) → flag, suggest reformatting.

### 8. Training loop

Compare to the canonical pattern (see `create_project_driver` for the template). Flag deviations:
- Missing `model.train()` / `model.eval()` toggling.
- Validation loop not wrapped in `with torch.no_grad():`.
- `optimizer.zero_grad()` forgotten.
- `tqdm(range(EPOCHS))` not used (or replaced with bare `range`).
- `cost.item()` accumulation missing → loss not detached, will retain graph.
- `train_cost[epoch] /= len(train_loader)` missing → unaveraged batch loss.
- Missing `tic = time.time()` / `toc = time.time()` and final `print(f"elapsed time {toc - tic:.2f} s")`.

### 9. Plotting

- Color used outside the semantic palette without justification:
  - black/`"k"` = training data, true function, primary prediction
  - red/`"r"` = validation, uncertainty envelope
  - blue/`"b"` = reference, alternative
  - `"cividis"` = loss landscape
  - `"hot_r"` (log-scaled) = error maps
- Cost format not `:.2e`, time format not `:.2f` → flag.
- Saved as `.png` for line/contour plots (should be `.pdf`) → flag. Saved as `.pdf` for raster (should be `.png`) → flag.

### 10. Comments and docstrings

- Docstrings in driver scripts → flag, remove.
- Comments describing **what** the code does → flag, remove (only "why" comments allowed when reason is non-obvious).
- Comments describing trivial behavior → flag.

### 11. Main guard

- `if __name__ == "__main__":` in a driver → flag, remove.

### 12. Init weights

- `init_weights(model, activation)` called before `.to(device)` → flag, swap order.
- `init_weights` missing entirely on a freshly created `nn.Module` → flag (suggest adding).

### 13. Saving outputs

- Manual CSV writing instead of `save_csv` from `postprocessing.py` → flag.
- Hardcoded paths in `fig.savefig(...)` → flag, use `RESULTS_DIR`.

## Reporting format

Group by section, file:line where possible. Conclude with a summary line.

```
## Paths and reproducibility
- mlp_expressivity.py:14 — missing `torch.manual_seed(...)` after device line
- mlp_from_scratch.py:67 — `torch.load("model.pt2").to(device)` → use `map_location=device`

## Aligned assignments
- mlp_expressivity.py:23–28 — aligned `=` block, collapse to single space

## Plotting
- ad.py:104 — line plot saved as `.png`; should be `.pdf`

## Section headers
- gabor_sgd.py:78 — training loop has no section header

Summary: 7 findings across 4 files. No auto-fix applied.
```

If a section has no findings, omit it from the report.

## Cross-checks

- For each file flagged, also confirm `python_hard_rules` rules are followed — this audit is the superset.
- Do **not** audit content style (comment wording, variable names) beyond the rules above. Style enforcement, not code review.
