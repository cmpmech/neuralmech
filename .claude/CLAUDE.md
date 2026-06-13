# NeuralMech — Claude Code Project Guide

## About
ML-enhanced physics solvers accompanying the book **"Deep Learning in Computational Mechanics"** (3rd ed., Springer).
Central question: *When and where is deep learning useful in numerical simulation?*

## Where the rules live

The detailed conventions live in skills under `.claude/skills/`. Invoke them by task:

| Skill | When |
|---|---|
| `python_hard_rules` | Editing or writing any `.py` file — must-follow paths, seeds, model loading, imports, no aligned `=`. |
| `unify_style` | Per-project audit (`projects/<N>_*/`) — report violations, suggest fixes. |
| `create_project_driver` | New driver script in `projects/`. Full 10-step template, plotting palette, save formats. |
| `create_unit_test` | Pytest tests for `NN.py`, `DL.py`, `ML.py`, `postprocessing.py` only. Drivers are not unit-tested. |
| `optimize_code` | Iterative improvement against a metric (wall clock, validation error, accuracy, memory). |
| `pvpython` | Writing or editing pvpython/ParaView render scripts in `projects/0_pvpython/` — render pipeline, point-cloud sphere glyphs (avoiding impostor speckles), the shared `.cmap/` colormaps + Spectral pitfall, supersampled AA, transparent backgrounds, legacy-VTK point export. |
| `mlhp` | Writing or editing any mlhp-based FEM driver — full Python API (meshes/grids, refinement, implicit CSG, hp/B-spline bases, fields, quadrature/FCM, elasticity/Poisson integrands, BCs, sparse + matrix-free assembly, solvers, postprocessing, numbering conventions) plus the `projects/16_elastic_fem/` voxel-FEM recipes (NumPy + CUDA matrix-free, CT geometry loading). |

## Setup

```bash
pip install -r requirements.txt
```

### Clone (with submodule)
```bash
git clone --recurse-submodules https://github.com/Leon-Herrmann/neuralmech
```

### Submodule: mlhp
- Path: `solvers/mlhp`
- Source: https://gitlab.com/hpfem/code/mlhp (pinned at `0.1.2`)
- Quick install: `pip install mlhp` (advanced physics requires C++ build)
- Init/update submodule: `git submodule update --init --recursive`

## Project structure

| Path | Description |
|---|---|
| `NN.py` | Network architectures (MLP, DCN, GNN variants, RNN, NODE, Bayesian, ResNet, ELM, AE/VAE, KAN) |
| `DL.py` | Deep learning utilities (weight init, Standardizer) |
| `ML.py` | ML utilities |
| `postprocessing.py` | Output helpers (save_csv, show_image) |
| `projects/` | Chapter-specific experiment drivers |
| `templates/` | Reusable patterns (e.g. training loop) |
| `solvers/` | Classical physics solvers + mlhp submodule |
| `models/` | Saved/trained networks |
| `data/` | Generated data (gitignored if large) |
| `external_data/` | Data generation tools (excluded from main repo) |
| `results/` | Post-processing outputs (gitignored) |
| `tests/` | Pytest tests for shared library files (created on demand) |

## Quick conventions (fallback cheat-sheet)

If the relevant skill is not loaded, these basics apply everywhere:

- **Paths**: `BASE_DIR = Path(__file__).parent` always; never `os.getcwd()`.
- **Reproducibility**: every PyTorch script needs `torch.manual_seed(<seed>)` and `torch.backends.cudnn.deterministic = True` after device setup.
- **Model loading**: `torch.load(..., map_location=device)` — never `.to(device)` afterward.
- **No aligned `=`**: single space on each side of `=`, never padded columns.
- **Driver flags**: `--book` (write to `results/`), `--animate` (write frames), no flag (interactive `plt.show()`).
- **No `__main__` guard** in drivers; modules execute top-to-bottom.

For the full ruleset, invoke `python_hard_rules`. For driver scaffolding, `create_project_driver`.

## Key library conventions

- All networks are `nn.Module` subclasses; take `layers`, `activations`, `normalizations` lists — length-matched to layer count.
- Graph networks: input is a PyG `Data` object (`graph.x`, `graph.edge_index`).
- `Standardizer` (`DL.py`): call `.inverse()` to undo normalization.
- `ELM` (`NN.py`): use `.fit(x, y, regularization)` before forward pass.

## Stack

- Python · PyTorch · torch-geometric · torchdiffeq · efficient-kan
- JAX/Flax, TensorFlow/Keras also installed
- Bayesian: PyMC, NumPyro, Pyro
- Solvers: mlhp, scipy, triangle
- Visualization: matplotlib, plotly, seaborn
- Testing: pytest

## Gitignored

`results/`, `results/animations/`, `data/`, `external_data/`, `.assets/original_images`, `.assets/animations`, `.idea/`, `projects/0_conceptual_figures`, `projects/0_helpers`
