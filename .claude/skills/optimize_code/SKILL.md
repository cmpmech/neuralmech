---
name: optimize_code
description: Iteratively improve a piece of code against a metric the user specifies. The user provides a target file (or function) and a metric to optimize — wall clock time, validation error, solution accuracy, peak memory, etc. — and this skill measures, modifies, and re-measures. Invoke when the user asks to "speed up", "tune", "optimize", or "improve the metric on" a specific piece of code. The user owns how invasive the changes are; ask if unclear.
---

Improve code against a user-specified metric. Measure → change → re-measure → report.

## What you need from the user

Before starting, confirm:

1. **Target**: which file, which function, which driver. An exact path.
2. **Metric**: how to measure. Wall clock seconds? Final `val_cost`? A solution-accuracy norm? Peak GPU memory?
3. **Baseline**: the current value of that metric (run once if not provided).
4. **Constraint**: how invasive. "Just hyperparameters." "Refactor the loss." "Anything that preserves the public API."
5. **Budget**: how many iterations or how much time before reporting back. Default: ~3 attempts.

If any of these is missing and not obvious, ask before changing code.

## Procedure

1. **Measure baseline.** Run the target with the current code. Record metric value, wall clock, any other relevant numbers (epoch count, GPU/CPU, etc.).
2. **Form a hypothesis.** One specific change you expect to improve the metric. Write it down before editing.
3. **Apply the change.** Edit the file. Keep the diff small and reviewable.
4. **Re-measure.** Run again. Compare to baseline.
5. **Decide**: keep, revert, or modify the change.
6. **Repeat** up to the budget.
7. **Report** the final metric, the diff that produced it, and any hypotheses that didn't work (one line each).

## Hypotheses to try, by metric

The user said "keep this minimal" — these are starting points, not exhaustive. Pick the cheapest plausible win first.

### Wall clock time

- Move tensors to GPU once, before the loop, not per iteration.
- Use `torch.compile(model)` (PyTorch 2.x) for the forward+backward pass.
- Check batch size — too small underutilizes the GPU, too large fragments memory.
- Profile with `torch.profiler` if the bottleneck is unclear.
- Replace Python-level loops with vectorized tensor ops.
- Disable `cudnn.deterministic` only if reproducibility is **not** required (rare in this repo — reproducibility is a hard rule, see `neuralmech-style`).
- For data loading: use `num_workers > 0`, `pin_memory=True`.
- Cache results that are recomputed each epoch.

### Validation error (training quality)

- Tune learning rate first — usually the largest single lever.
- Add or adjust weight decay (`REGULARIZATION`).
- Try a different optimizer: SGD+momentum vs. AdamW vs. LBFGS.
- Try a different activation; for MLPs, Tanh vs. ReLU vs. GELU vs. SIREN often shifts results materially.
- Increase capacity (width/depth) if underfitting; reduce or add regularization if overfitting.
- Check standardization — `Standardizer` on inputs and outputs typically helps.
- Increase `EPOCHS` only if the loss is still decreasing at the end.
- Add early stopping with patience to avoid noise from late-epoch divergence.

### Solution accuracy (numerical)

- Tighten solver tolerance (mlhp / scipy options).
- Refine the discretization (mesh, quadrature, time step).
- Check unit/dimension consistency in physics terms.
- Increase polynomial order in `mlhp` if applicable.
- Replace the loss with a stronger norm (e.g., $L^2$ → $H^1$).

### Peak memory

- Reduce batch size.
- Use `with torch.no_grad():` in any path that doesn't need gradients.
- Free intermediates explicitly (`del x; torch.cuda.empty_cache()`) inside long loops.
- Use gradient checkpointing for very deep models (`torch.utils.checkpoint.checkpoint`).
- Switch to mixed-precision (`torch.amp.autocast`) — confirm the accuracy is acceptable.

## Measurement helpers

### Wall clock

```python
import time
tic = time.time()
# ... code under test ...
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")
```

Already standard in drivers. Run multiple times if the variance matters.

### Final validation cost

Take `val_cost[-1]` after training. If early stopping kicks in, take `min(val_cost)` instead.

### Peak GPU memory

```python
import torch
torch.cuda.reset_peak_memory_stats()
# ... code under test ...
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"peak GPU memory: {peak:.2f} GB")
```

## Reporting format

After the budget runs out:

```
## Baseline
- wall clock: 42.13 s
- val_cost[-1]: 1.34e-3

## Final
- wall clock: 18.67 s   (-56%)
- val_cost[-1]: 1.21e-3 (-10%)

## Diff
[unified diff of changes]

## Hypotheses tried
- ✓ moved standardizer init outside loop  → -8s
- ✓ batch size 64 → 128                     → -15s, no quality loss
- ✗ torch.compile(model)                    → silent fallback on this PyTorch version, no win
- ✓ AdamW → SGD+momentum                    → minor val_cost improvement
```

## Boundaries

- **Don't change the experiment's intent.** If the driver is comparing two architectures, don't unify them; if it sweeps a hyperparameter, don't fix the value.
- **Don't break the `--book` / `--animate` outputs.** A driver that feeds a book figure must remain reproducible against the printed figure (see the monorepo CLAUDE.md). If a change shifts the output, flag it before keeping the change.
- **Don't bypass the `neuralmech-style` hard rules.** The reproducibility lines, `Path(__file__).parent`, etc. stay even when optimizing.
- **One change per iteration.** Don't bundle multiple changes — you lose attribution when something works (or breaks).
