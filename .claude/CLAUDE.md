# NeuralMech — Claude Code Project Guide

## About
ML-enhanced physics solvers accompanying the book **"Deep Learning in Computational Mechanics"** (3rd ed., Springer).
Central question: *When and where is deep learning useful in numerical simulation?*

## Where the rules live

The detailed conventions live in skills under `.claude/skills/`. Invoke them by task:

| Skill | When |
|---|---|
| `neuralmech-style` | Writing or editing ANY `.py` in `code/` — the authoritative rules and house style. |
| `mlhp-overview` | Any mlhp work, C++ or Python; it routes on to `mlhp` (full Python API + the `projects/18_mlhp/` voxel-FEM recipes), `mlhp-immersed-methods`, `mlhp-visualize`, `mlhp-cpp-style`, `mlhp-cpp-utilities`. |
| `pvpython` | pvpython/ParaView render scripts in `projects/0_pvpython/`. |
| `sweep-dump` | Sweeping a driver over a few constants and dumping samples for the user to judge. |
| `optimize_code` | Iterative improvement against a metric (wall clock, error, memory). |

## Setup

`pip install -r requirements.txt`, then `git submodule update --init --recursive`.

### Submodule: mlhp
- Path: `solvers/mlhp_source/mlhp`, source https://gitlab.com/hpfem/code/mlhp, pinned at `0.2.4`
- Build: `cmake -S solvers/mlhp_source -B solvers/mlhp_build && cmake --build solvers/mlhp_build -j`
- The venv `.pth` puts `solvers/mlhp_build/bin` on `sys.path`, so `import mlhp` resolves to the
  compiled package (`bin/mlhp/`, NeuralMech helper integrands bundled into `mlhp._core`), **not**
  the PyPI `mlhp`. `pip install mlhp` works but lacks the advanced physics.

### Dependency: cuwave
GPU finite difference wave solver and its adjoints (https://github.com/cmpmech/cuwave). Pip-installed
and pinned in `requirements.txt` — not a submodule, there is no `solvers/cuwave` checkout. Needs a
cupy matching the CUDA toolkit. For who uses it: `grep -rl cuwave projects/`.

## Layout

`NN.py` architectures · `DL.py` weight init + Standardizer · `ML.py` · `postprocessing.py` save_csv/show_image
· `projects/<chapter>_<name>/` chapter drivers · `solvers/` classical solvers + mlhp.

`external_data/` holds data-generation tools excluded from the main repo. `data/`, `models/` and
`results/` are gitignored — regenerate rather than expecting contents.

## Quick conventions (fallback cheat-sheet)

If the relevant skill is not loaded, these basics apply everywhere:

- **Paths**: `BASE_DIR = Path(__file__).parent` always; never `os.getcwd()`.
- **Reproducibility**: every PyTorch script needs `torch.manual_seed(<seed>)` and `torch.backends.cudnn.deterministic = True` after device setup.
- **Model loading**: `torch.load(..., map_location=device)` — never `.to(device)` afterward.
- **No aligned `=`**: single space on each side of `=`, never padded columns.
- **Driver flags**: `--book` (write to `results/`), `--animate` (write frames), no flag (interactive `plt.show()`).
- **No `__main__` guard** in drivers; modules execute top-to-bottom.

For the full ruleset, driver style, and scaffolding, invoke `neuralmech-style`.

## Key library conventions

- All networks are `nn.Module` subclasses. The Sequential-family builders (`MLP`, `DCN`, `BayesianMLP`) take `layers`/`channels` plus per-layer `post_modules` (applied after each core layer; activations live here by default) and optional `pre_modules` (applied before; resampling/pre-norm). Each entry is `None`, a module, or a list of modules; `None` entries are skipped. `post_modules` is the 2nd positional arg, so `MLP(LAYERS, ACTIVATIONS)` still reads correctly.
- Graph networks: input is a PyG `Data` object (`graph.x`, `graph.edge_index`).
- `Standardizer` (`DL.py`): call `.inverse()` to undo normalization.
- `ELM` (`NN.py`): use `.fit(x, y, regularization)` before forward pass.
