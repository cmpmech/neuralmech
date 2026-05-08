---
name: create_unit_test
description: Write pytest unit tests for the four shared library files in `code/` — `NN.py` (architectures), `DL.py` (training utilities), `ML.py` (ML utilities), `postprocessing.py` (output helpers). Bootstraps `code/tests/` on first invocation if it doesn't exist. Invoke when the user asks to add tests for a class/function in those files. Drivers in `projects/` are experiment scripts and are NOT unit-tested.
---

Add pytest unit tests for the shared library files in `code/`. Drivers (`projects/`) are not in scope — they are one-off experiment scripts whose correctness is validated by their figure output.

## In scope

| File | What to test |
|---|---|
| `NN.py` | Architecture classes — forward pass shape, parameter count, `nn.Module` invariants, special methods (e.g., `ELM.fit`, `BayesianMLP.kl_divergence`, `ResNet` skip indices, `SIREN.omega_0` scaling) |
| `DL.py` | `init_weights` (per-activation behavior), `Standardizer` / `Normalizer` (forward + `.inverse()` round-trip), `get_params` / `set_params` / `flatten_params` round-trips, `filter_normalize_direction` shape, `build_ae_cnn_config` symmetry |
| `ML.py` | ML utility functions (read the file to enumerate testable surfaces) |
| `postprocessing.py` | `save_csv` (column names match kwargs, content matches arrays), `show_image` (file written when `path` given) |

## Out of scope

- `projects/` drivers — no tests.
- `solvers/` — has its own structure; only test if the user explicitly asks.
- `templates/` — patterns, not testable units.

## Test layout

If `code/tests/` does not exist, bootstrap on first invocation:

```
code/
  tests/
    __init__.py            # empty
    conftest.py            # shared fixtures (device, fixed seed)
    test_NN.py
    test_DL.py
    test_ML.py
    test_postprocessing.py
```

One test file per library module. Group tests by class/function inside the file.

### `conftest.py` skeleton

```python
import pytest
import torch


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def device():
    return torch.device("cpu")   # tests run on cpu for portability
```

Tests run on CPU regardless of GPU availability — guarantees determinism and CI portability.

## Style

Tests follow pytest idioms, **not** the driver style guide. Differences:

- `def test_<thing>():` functions, no `__main__` guard.
- Docstrings allowed but not required; one short line if the test name is ambiguous.
- `assert` over `unittest`'s `self.assertX`.
- Use fixtures for shared setup (device, seed, sample tensors).
- Use `pytest.mark.parametrize` for tabulated cases (different activations, dimensions, etc.).
- File naming: `test_<module>.py`; function naming: `test_<feature>_<condition>`.
- Aligned `=` is still **not** used (project-wide rule applies in tests too).
- No printing in tests — assertions only.

## What to test

For each public class/function, cover at minimum:

1. **Construction succeeds** with a valid arg set.
2. **Construction fails** (or warns) with a clearly invalid arg set, where the failure mode is documented.
3. **Forward pass shape** (for `nn.Module`) — input → output dimensions match expectation.
4. **Round-trip behavior** where applicable: `Standardizer(x).inverse(Standardizer(x)(x)) == x`, `unflatten_params(flatten_params(p)) == p`.
5. **Determinism**: with the autouse seed fixture, two constructions yield identical parameters.
6. **Edge cases**: zero-length input, single-sample batch, single-class output.

## Templates

### `nn.Module` forward shape (e.g., MLP)

```python
import pytest
import torch
import torch.nn as nn

from NN import MLP


@pytest.mark.parametrize("layers,activations", [
    ([1, 16, 1], [nn.Tanh(), None]),
    ([4, 32, 32, 2], [nn.ReLU(), nn.ReLU(), None]),
])
def test_mlp_forward_shape(layers, activations):
    model = MLP(layers, activations)
    x = torch.randn(8, layers[0])
    y = model(x)
    assert y.shape == (8, layers[-1])


def test_mlp_parameter_count():
    model = MLP([2, 4, 1], [nn.Tanh(), None])
    expected = (2 * 4 + 4) + (4 * 1 + 1)   # weights + biases
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected
```

### Round-trip (e.g., Standardizer)

```python
import torch
from DL import Standardizer


def test_standardizer_round_trip():
    x = torch.randn(100, 3)
    s = Standardizer(x, dim=0)
    z = s(x)
    x_rec = s.inverse(z)
    assert torch.allclose(x_rec, x, atol=1e-6)


def test_standardizer_zero_mean_unit_var():
    x = torch.randn(1000, 2)
    s = Standardizer(x, dim=0)
    z = s(x)
    assert torch.allclose(z.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(z.std(dim=0, unbiased=False), torch.ones(2), atol=1e-5)
```

### `init_weights` per-activation (e.g., DL.py)

```python
import torch.nn as nn

from NN import MLP
from DL import init_weights


def test_init_weights_zeros_biases():
    model = MLP([2, 4, 1], [nn.Tanh(), None])
    init_weights(model, nn.Tanh())
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            assert (layer.bias == 0).all()
```

### `save_csv` column matching (postprocessing.py)

```python
import numpy as np

from postprocessing import save_csv


def test_save_csv_columns_match_kwargs(tmp_path):
    path = tmp_path / "out.csv"
    save_csv(path, x=np.array([1, 2, 3]), y=np.array([4.0, 5.0, 6.0]))
    header = path.read_text().splitlines()[0]
    assert "x" in header and "y" in header
```

### Parametrized failures

```python
import pytest
from NN import MLP


@pytest.mark.parametrize("layers,activations", [
    ([], []),
    ([1], []),
    ([1, 2, 1], [None]),   # too few activations
])
def test_mlp_invalid_construction(layers, activations):
    with pytest.raises((ValueError, IndexError, AssertionError)):
        MLP(layers, activations)
```

## How to invoke

When the user asks for tests on a specific class or function:

1. Read the source file to see the actual signature, docstring, and special methods.
2. If `code/tests/` doesn't exist, create it with `__init__.py`, `conftest.py`, and the relevant `test_<module>.py`.
3. If `test_<module>.py` exists, append new test functions in the same file.
4. Run `pytest code/tests/test_<module>.py -v` to verify the new tests pass.
5. Report which tests pass/fail.

## Running

From `code/`:

```bash
pytest tests/ -v
pytest tests/test_NN.py -v
pytest tests/test_NN.py::test_mlp_forward_shape -v
```

If tests fail because of a missing dependency (e.g., `torch-geometric` not installed in the active env), report which test was skipped and why; do not silently mark it passing.
