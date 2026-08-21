"""Hyperparameter search over the 1D neural reparametrization of reparametrization1D.py.

Answers two questions with one Optuna study: which combination reparametrizes the
landscape best, and which hyperparameters that outcome actually depends on. The second is
the interesting one -- a reparametrization that only works for one hand-tuned setting is
not evidence that reparametrization helps.

Add, remove or pin an axis by editing SEARCH_SPACE alone. A tuple is a searched axis; a
plain value is a constant, which drops the name out of the study and therefore out of the
importance analysis.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.importance import (
    FanovaImportanceEvaluator,
    PedAnovaImportanceEvaluator,
    get_param_importances,
)
from optuna.samplers import QMCSampler, RandomSampler, TPESampler

from reparametrization_config import GUESS, RANGE, Linear, MLP, optimize

# -------------------------------------- settings -------------------------------------
# search space: ("float", lo, hi[, "log"]) | ("int", lo, hi) | ("cat", [...]) | a constant
SEARCH_SPACE = {
    "lr": ("float", 1e-3, 1e0, "log"),
    "depth": ("int", 1, 4),
    "width": ("int", 4, 64),
    "activation": ("cat", ["GELU", "SiLU", "ReLU", "Tanh", "Sigmoid"]),
    "input_dim": ("int", 1, 64),
    "input_scale": ("float", 1e-2, 1e1, "log"),
    "learnable": ("cat", [False, True]),
    "scheme": ("cat", ["default", "matched", "orthogonal", "xavier"]),
    "gain": ("float", 1e-1, 5e0, "log"),
    "output_activation": ("cat", [True, False]),
}

# search
N_TRIALS = 300
SAMPLER = "tpe"  # "tpe" finds the best combination, "qmc"/"random" rank importance fairly
STORAGE = None  # e.g. "sqlite:///reparametrization1D_tune.db" to resume or use the dashboard

# optimization
EPOCHS = 50
LR_LINEAR = 2e-1  # the plain-descent baseline, from reparametrization1D.py

# initialization (results should be consistent if this is changed)
SEED = 1
SEEDS = 5  # paired repeats averaged per trial
SEEDS_BEST = 50  # repeats used to profile the winner

# postprocessing
COLUMNS = 3

# keys consumed by optimize() rather than by the model
OPTIMIZE_KEYS = {"lr", "epochs", "optimizer"}


np.random.seed(0)
torch.manual_seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)


# ------------------------------------ search space -----------------------------------
def suggest(trial, name, spec):
    """Register one axis with the trial; a non-tuple spec is a constant, not an axis."""
    if not isinstance(spec, tuple):
        return spec

    kind, *rest = spec
    if kind == "float":
        lo, hi, *log = rest
        return trial.suggest_float(name, lo, hi, log=bool(log))
    if kind == "int":
        return trial.suggest_int(name, *rest)
    if kind == "cat":
        return trial.suggest_categorical(name, rest[0])
    raise ValueError(f"unknown axis kind {kind!r}")


def is_log(spec):
    return isinstance(spec, tuple) and spec[0] == "float" and "log" in spec


AXES = [name for name, spec in SEARCH_SPACE.items() if isinstance(spec, tuple)]


# ------------------------------------- evaluation ------------------------------------
def evaluate(params, seeds):
    """Return the final (x, f(x)) of `seeds` repeats of the reparametrized descent.

    The seed is reset per repeat rather than left to drift, so every configuration sees
    the same sequence of networks. With only a handful of repeats that pairing is what
    keeps seed noise from swamping the hyperparameter effect.
    """
    model_kwargs = {k: v for k, v in params.items() if k not in OPTIMIZE_KEYS}
    optimize_kwargs = {k: v for k, v in params.items() if k in OPTIMIZE_KEYS}
    optimize_kwargs.setdefault("epochs", EPOCHS)

    finals = np.zeros((seeds, 2))
    for seed in range(seeds):
        torch.manual_seed(SEED + seed)
        model = MLP(GUESS, **model_kwargs)
        finals[seed] = optimize(model, **optimize_kwargs)[-1]
    return finals[:, 0], finals[:, 1]


def objective(trial):
    params = {name: suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()}
    _, ys = evaluate(params, SEEDS)
    return float(ys.mean())


# --------------------------------------- search --------------------------------------
sampler = {
    "tpe": lambda: TPESampler(seed=SEED, multivariate=True),  # lr/scale/gain interact
    "random": lambda: RandomSampler(seed=SEED),
    "qmc": lambda: QMCSampler(seed=SEED),
}[SAMPLER]()

study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    storage=STORAGE,
    study_name="reparametrization1D",
    load_if_exists=STORAGE is not None,
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# the baseline the reparametrization has to beat
torch.manual_seed(SEED)
history_linear = optimize(Linear(GUESS), LR_LINEAR, EPOCHS)
x_linear, y_linear = history_linear[-1]

print(f"\nplain descent      f = {y_linear:.4f}  at x = {x_linear:+.4f}")
print(f"reparametrization  f = {study.best_value:.4f}  (mean over {SEEDS} seeds)")
print("\nbest configuration")
for name, value in study.best_params.items():
    print(f"  {name:20s} {value}")

# ------------------------------------- importance ------------------------------------
# TPE concentrates trials near the optimum, which biases fANOVA's variance decomposition.
# PedANOVA is built for that regime, so agreement between the two is the trust signal and
# disagreement is the cue to re-run with SAMPLER = "qmc".
importances = {
    "fANOVA": get_param_importances(
        study, evaluator=FanovaImportanceEvaluator(seed=SEED)
    ),
    "PedANOVA": get_param_importances(study, evaluator=PedAnovaImportanceEvaluator()),
}

ranking = sorted(
    AXES, key=lambda n: -np.mean([i.get(n, 0.0) for i in importances.values()])
)

print(f"\n{'hyperparameter':<20}{'fANOVA':>10}{'PedANOVA':>10}")
for name in ranking:
    fanova, pedanova = (importances[k].get(name, 0.0) for k in importances)
    print(f"{name:<20}{fanova:>10.3f}{pedanova:>10.3f}")

# ----------------------------------- postprocessing ----------------------------------
values = np.array([t.value for t in study.trials if t.value is not None])
numbers = np.array([t.number for t in study.trials if t.value is not None])
xs_best, ys_best = evaluate({**SEARCH_SPACE, **study.best_params}, SEEDS_BEST)
inside = np.abs(xs_best) <= max(abs(RANGE[0]), abs(RANGE[1]))

# the winner is tuned on SEEDS seeds, so check it on more: a mean far above the median
# means a few seeds fail outright rather than the configuration being uniformly mediocre
print(f"\nbest configuration over {SEEDS_BEST} seeds")
print(f"  mean   f = {ys_best.mean():.4f}")
print(f"  median f = {np.median(ys_best):.4f}")
print(f"  beat plain descent in {int((ys_best < y_linear).sum())}/{SEEDS_BEST} seeds")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

ax[0].scatter(numbers, values, c=numbers, cmap="cividis", s=12)
ax[0].plot(numbers, np.minimum.accumulate(values), "k", lw=1.5, label="best so far")
ax[0].axhline(y_linear, color="r", lw=1.2, ls="--", label="plain descent")
ax[0].set(xlabel="trial", ylabel="mean final $f(x)$", title=f"search ({SAMPLER})")
ax[0].set_yscale("log")  # a diverged trial reaches f ~ 1e5 and flattens everything else
ax[0].legend()

y = np.arange(len(ranking))[::-1]
colors = plt.get_cmap("cividis")([0.25, 0.7])
for offset, (label, color) in zip((-0.2, 0.2), zip(importances, colors)):
    scores = [importances[label].get(name, 0.0) for name in ranking]
    ax[1].barh(y + offset, scores, 0.4, color=color, label=label)
ax[1].set(yticks=y, yticklabels=ranking, xlabel="importance", title="what matters")
ax[1].legend()

ax[2].hist(xs_best, bins=np.linspace(RANGE[0], RANGE[1], 41), color="b")
ax[2].axvline(x_linear, color="r")
ax[2].set(
    xlim=RANGE,
    xlabel="final $x$",
    ylabel="seeds",
    # seeds that ran outside RANGE are not drawn, so say how many
    title=f"best configuration, {inside.sum()}/{SEEDS_BEST} seeds in range",
)

fig.tight_layout()

# marginals: the honest check on any importance ranking
rows = -(-len(ranking) // COLUMNS)
fig2, ax2 = plt.subplots(rows, COLUMNS, figsize=(4 * COLUMNS, 3 * rows), squeeze=False)
for axis, name in zip(ax2.ravel(), ranking):
    drawn = [(t.params[name], t.value) for t in study.trials if name in t.params]
    raw, scores = zip(*drawn)
    spec = SEARCH_SPACE[name]

    if spec[0] == "cat":
        levels = list(spec[1])
        position = np.array([levels.index(v) for v in raw], dtype=float)
        position += np.random.uniform(-0.15, 0.15, position.shape)  # jitter to see density
        axis.set(xticks=range(len(levels)), xticklabels=[str(v) for v in levels])
    else:
        position = np.array(raw, dtype=float)
        if is_log(spec):
            axis.set_xscale("log")

    axis.scatter(position, scores, c="k", s=8, alpha=0.4)
    axis.set(xlabel=name, ylabel="mean final $f(x)$", yscale="log")
    axis.axhline(y_linear, color="r", lw=1.0, ls="--")

for axis in ax2.ravel()[len(ranking) :]:
    axis.axis("off")

fig2.tight_layout()
plt.show()
