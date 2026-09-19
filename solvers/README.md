# Solvers

Classical physics solvers the drivers in [`projects/`](../projects/README.md) build on.

## External solvers

- [cuwave](https://github.com/cmpmech/cuwave): differentiable GPU finite difference wave solver, behind the wave and transient acoustic topology optimization drivers. Installed from PyPI with `pip install cuwave`.
- `mlhp_source/mlhp`: _hp_ finite element kernel. `pip install mlhp` covers the Python
  API; advanced physics needs the C++ build (see below).

### Building mlhp from source

The PyPI wheel ships the core Python API only. The NeuralMech helper integrands in
`mlhp_source/helpers/` (custom quadrature and material integrands) are C++ and have to be
compiled together with mlhp into a single `mlhp` package. Needs CMake >= 3.12 and a C++20
compiler.

```
git submodule update --init --recursive
cmake -S solvers/mlhp_source -B solvers/mlhp_build
cmake --build solvers/mlhp_build -j
```

Then put the build on `sys.path` ahead of the PyPI package with a `.pth` file in the
environment's `site-packages` (run from `code/`, inside the activated environment):

```
echo "import sys; sys.path.insert(0, '$PWD/solvers/mlhp_build/bin')" > "$(python -c 'import site; print(site.getsitepackages()[0])')/mlhp_build.pth"
```

`import mlhp` now resolves to `solvers/mlhp_build/bin/mlhp/`, with the helper integrands
bundled into `mlhp._core`.

## Modules

| file                | description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| `optimization.py`   | MMA, structured-grid FEM assemble-and-solve, conic density filter, Heaviside projection |
| `homogenization.py` | KUBC energy homogenization of a two-phase cell                                          |
| `truss.py`          | linear-elastic truss with a global stiffness matrix                                     |
| `dynamic_mdof.py`   | multi-degree-of-freedom spring-mass-damper chain                                        |
| `multibody.py`      | planar multibody trees integrated through their Lagrangian equations                    |
| `bouncing_balls.py` | impulse-based ball contact under gravity                                                |
| `mklwrapper.py`     | MKL pardiso factorization bindings                                                      |
