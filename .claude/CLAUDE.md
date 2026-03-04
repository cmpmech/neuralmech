# NeuralMech — Claude Code Project Guide

## About
ML-enhanced physics solvers accompanying the book **"Deep Learning in Computational Mechanics"** (3rd ed., Springer).
Central question: *When and where is deep learning useful in numerical simulation?*

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

## Project Structure

| Path | Description |
|---|---|
| `NN.py` | Network architectures (MLP, DCN, GNN variants, RNN, NODE, Bayesian, ResNet, ELM, AE/VAE, KAN) |
| `DL.py` | Deep learning utilities (weight init, Standardizer) |
| `ML.py` | ML utilities |
| `projects/` | Chapter-specific experiment drivers |
| `templates/` | Reusable patterns (e.g. training loop) |
| `solvers/` | Classical physics solvers + mlhp submodule |
| `models/` | Saved/trained networks |
| `data/` | Generated data (gitignored if large) |
| `external_data/` | Data generation tools (excluded from main repo) |
| `results/` | Post-processing outputs (gitignored) |

## Key Conventions

- **Architecture style**: All networks are `nn.Module` subclasses; take `layers`, `activations`, `normalizations` lists — length-matched to layer count
- **Graph networks**: Input is a PyG `Data` object (`graph.x`, `graph.edge_index`)
- **Standardizer** (`DL.py`): call `.inverse()` to undo normalization
- **ELM** (`NN.py`): use `.fit(x, y, regularization)` before forward pass

## Stack
- Python · PyTorch · torch-geometric · torchdiffeq · efficient-kan
- JAX/Flax, TensorFlow/Keras also installed
- Bayesian: PyMC, NumPyro, Pyro
- Solvers: mlhp, scipy, triangle
- Visualization: matplotlib, plotly, seaborn

## Gitignored
`results/`, `results/animations/`, `data/`, `external_data/`, `.assets/original_images`, `.assets/animations`, `.idea/`, `projects/0_conceptual_figures`, `projects/0_helpers`
